import numpy as np
import math
import torch
import torch.nn as nn
from collections import Counter
from scipy.stats import mode
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add, scatter_max, scatter_mean

class PrecursorClassifier(nn.Module):
    """
    The Roost model is comprised of a fully connected network
    and message passing graph layers.
    The message passing layers are used to determine a descriptor set
    for the fully connected network. The graphs are used to represent
    the stoichiometry of inorganic materials in a trainable manner.
    This makes them systematically improvable with more data.
    """

    def __init__(
        self,
        task,
        pooling,
        globalfactor,
        gru,
        device,
        robust,
        n_targets,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        out_hidden=[1024, 512, 256, 128, 64],
        #**kwargs
    ):
        super().__init__()

        desc_dict = {
            "pooling": pooling,
            "globalfactor": globalfactor,
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
        }

        self.material_nn = AtomicDescriptorNetwork(**desc_dict)
        self.robust = robust
        self.pooling = pooling
        self.globalfactor = globalfactor
        self.gru = gru

        # define an output neural network
        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets
        
        if gru == False:
            self.output_nn = ResidualNetwork(elem_fea_len, output_dim, out_hidden)
        else:
            if self.globalfactor:
                self.output_nn = GRU_Classifier(elem_fea_len*2, 512, output_dim, len(out_hidden), False, pooling, device)
            else:
                self.output_nn = GRU_Classifier(elem_fea_len, 512, output_dim, len(out_hidden), False, pooling, device)
        if (gru == False) and (pooling ==True):
            raise NotImplementedError('No gru=False & pooling=True mode')
        
    def forward(self, input_tar, metal_mask, source_elem_idx, pre_set_idx):
        """
        Forward pass through the material_nn and output_nn
        
        *input_tar
        ------------
        [0] : elem_weights
        [1] : elem_fea
        [2] : self_fea_idx
        [3] : nbr_fea_idx
        [4] : cry_elem_idx
        
        *input_pre
        ------------
        [0] : elem_weights
        [1] : elem_fea
        [2] : self_fea_idx
        [3] : nbr_fea_idx
        [4] : cry_elem_idx
        
        """
        if self.pooling == False:
            tar_elem_fea = self.material_nn(*input_tar, metal_mask, source_elem_idx)
            source_elem_fea = tar_elem_fea
        else:
            tar_elem_fea, source_elem_fea = self.material_nn(*input_tar, metal_mask, source_elem_idx)

        # apply neural network to map from learned features to target
        if self.gru == False:
            return self.output_nn(tar_elem_fea), source_elem_fea
        else:
            return self.output_nn(tar_elem_fea, pre_set_idx), source_elem_fea

    def __repr__(self):
        return self.__class__.__name__

class AtomicDescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        pooling,
        globalfactor,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
    ):
        """
        """
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)
        # self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                )
                for i in range(n_graph)
            ]
        )
        # define a global pooling function for materials
        self.pooling = pooling
        self.globalfactor = globalfactor
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
                )
                for _ in range(cry_heads)
            ]
        )

        # self.cry_pool = nn.ModuleList(
        #     [
        #         MeanPooling()
        #     ]
        # )

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, metal_mask, source_elem_idx):
        """
        Forward pass
        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch
        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx
        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)
        
        elem_fea_init = elem_fea
        elem_fea_init = elem_fea_init[torch.where(metal_mask!=-1)[0]]
        elem_fea_init = scatter_mean(elem_fea_init, source_elem_idx, dim=0)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)
        
        if self.pooling == False:
            
            if self.globalfactor == True:
                head_fea = []
                for attnhead in self.cry_pool:
                    head_fea.append(
                        attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
                        # attnhead(elem_fea, index=cry_elem_idx)
                    )
                global_agg_fea = torch.mean(torch.stack(head_fea), dim=0)
                cry_elem_idx = cry_elem_idx[torch.where(metal_mask!=-1)[0]]
                cry_elem_idx = scatter_mean(cry_elem_idx, source_elem_idx, dim=0)
                global_agg_elem_fea = []
                for i in range(len(cry_elem_idx)):
                    global_agg_elem_fea.append(global_agg_fea[cry_elem_idx[i]])
                global_agg_elem_fea = torch.stack(global_agg_elem_fea)
                
                source_elem_fea = torch.cat((global_agg_elem_fea, elem_fea_init), dim=1)
            else:
                # remove non-metal elem_fea & average metal elem_fea in the same crystal (composite)
                source_elem_fea = elem_fea[torch.where(metal_mask!=-1)[0]]
                source_elem_fea = scatter_mean(source_elem_fea, source_elem_idx, dim=0)
            
            return source_elem_fea
    
        else:
            # generate crystal features by pooling the elemental features
            source_elem_fea = elem_fea[torch.where(metal_mask!=-1)[0]]
            source_elem_fea = scatter_mean(source_elem_fea, source_elem_idx, dim=0)
            
            head_fea = []
            for attnhead in self.cry_pool:
                head_fea.append(
                    attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
                    # attnhead(elem_fea, index=cry_elem_idx)
                )
    
            return torch.mean(torch.stack(head_fea), dim=0), source_elem_fea
    

    def __repr__(self):
        return self.__class__.__name__

class GRU_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, pooling, device):
        super().__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        self.pooling = pooling
        
        self.gru = nn.GRU(input_size=self.input_size, hidden_size = self.hidden_size, num_layers=self.num_layers, bidirectional=bidirectional, batch_first=True)
        self.output_fc = nn.Linear(self.hidden_size * self.num_directions, output_size)

    def forward(self, x, pre_set_idx):
        if self.pooling:
            x_size = x.size()
            x = x.view([x_size[0],1,-1])
            max_seq_len = mode(pre_set_idx.cpu()).count.item()
            for i in range(max_seq_len):
                if i == 0:
                    x_elem = x
                else:
                    x_elem = torch.cat((x_elem,x), dim=1)
            # x.shape == (batch_sourceElem_size, 1, fea_len)
            # x_elem.shape == (batch_cry_size, seq_len, fea_len)
            
            cnt = Counter(pre_set_idx.cpu().tolist())
            self.batch_size = x_elem.size(0)
            pad_mask = []
            for i in range(self.batch_size):
                pad_mask += [1]*cnt[i] + [0]*(max_seq_len-cnt[i])
            
            h0 = self.init_hidden(self.device)
            output_elem, hn = self.gru(x_elem, h0)
            output = output_elem.reshape([-1,self.hidden_size]) 
            output = output[torch.where(torch.tensor(pad_mask)==1)[0]]
            
            output = self.output_fc(output)
        else:
            x_size = x.size()
            x = x.view([x_size[0],1,-1])
            # x.shape == (batch_sourceElem_size, seq_len, fea_len)
            
            self.batch_size = x.size(0)
            h0 = self.init_hidden(self.device)
            output, hn = self.gru(x, h0)
            
            output = self.output_fc(output[:, -1, :])
        return output
    
    def init_hidden(self, device):
        return torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).to(device)

class TemperatureRegressor(nn.Module):
    """
    The Roost model is comprised of a fully connected network
    and message passing graph layers.
    The message passing layers are used to determine a descriptor set
    for the fully connected network. The graphs are used to represent
    the stoichiometry of inorganic materials in a trainable manner.
    This makes them systematically improvable with more data.
    """

    def __init__(
        self,
        task,
        pooling,
        globalfactor,
        device,
        robust,
        n_targets,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        out_hidden=[1024, 512, 256, 128, 64],
        #**kwargs
    ):
        super().__init__()

        desc_dict = {
            "pooling": pooling,
            "globalfactor": globalfactor,
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
        }

        self.material_nn_tar = AtomicDescriptorNetwork(**desc_dict)
        self.material_nn_pre = AtomicDescriptorNetwork(**desc_dict)
        self.robust = robust

        # define an output neural network
        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets
        
        self.output_nn = ResidualNetwork(elem_fea_len*2, output_dim, out_hidden)
        
    def forward(self, input_tar, input_pre):
        """
        Forward pass through the material_nn and output_nn
        
        *input_tar
        ------------
        [0] : elem_weights
        [1] : elem_fea
        [2] : self_fea_idx
        [3] : nbr_fea_idx
        [4] : cry_elem_idx
        
        *input_pre
        ------------
        [0] : elem_weights
        [1] : elem_fea
        [2] : self_fea_idx
        [3] : nbr_fea_idx
        [4] : cry_elem_idx
        
        """
        tar_cry_fea, _ = self.material_nn_tar(*input_tar)
        pre_cry_fea, _ = self.material_nn_pre(*input_pre)
    
        syn_descriptor = torch.cat([tar_cry_fea, pre_cry_fea], dim=1)

        # apply neural network to map from learned features to target
        return self.output_nn(syn_descriptor), syn_descriptor

    def __repr__(self):
        return self.__class__.__name__

class LinearRegressor(nn.Module):
    def __init__(self, d_in,d_h,d_out, layer_num): #dimension_input, dimension_hidden, dimension_output, layer_num
        super().__init__()
        self.linear1 = nn.Linear(d_in,d_h)
        self.linear2 = nn.Linear(d_h,d_h)
        self.linear3 = nn.Linear(d_h,d_out)
        self.bn1 = nn.BatchNorm1d(d_h)
        self.relu = nn.ReLU()
        self.layer_num = layer_num

    def forward(self, x):
        retval = self.linear1(x)
        #retval = self.bn1(retval)
        retval = self.relu(retval)
        
        for i in range(self.layer_num):
            retval = self.linear2(retval)
            #retval = self.bn1(retval)
            retval = self.relu(retval)
        
        retval = self.linear3(retval)
        return retval

class LinearRegressor2(nn.Module):
    def __init__(self, d_in,d_h,d_out, layer_num): #dimension_input, dimension_hidden, dimension_output, 
        super().__init__()
        self.linear1 = nn.Linear(d_in,d_h)
        self.linear2 = nn.Linear(d_h,d_h)
        self.linear3 = nn.Linear(d_h+d_h,d_h+d_h)
        self.linear4 = nn.Linear(d_h+d_h,d_out)
        self.bn1 = nn.BatchNorm1d(d_h)
        self.relu = nn.ReLU()
        self.layer_num = layer_num

    def forward(self, x):
        retval = self.linear1(x)
        #retval = self.bn1(retval)
        retval = self.relu(retval)
        
        for i in range(int((self.layer_num)/2)):
            retval = self.linear2(retval)
            #retval = self.bn1(retval)
            retval = self.relu(retval)
        
        retval = torch.cat([retval[:,0,:],retval[:,1,:]], dim=1)
        retval = self.linear3(retval)
        retval = self.relu(retval)
        retval = self.linear3(retval)
        retval = self.relu(retval)
        
        retval = self.linear4(retval)
        return retval

class DenseLayer(nn.Module):
    def __init__(self, in_fea_len,out_fea_len):
        super(DenseLayer, self).__init__()
        self.lin1 = nn.Linear(in_fea_len,out_fea_len)
        self.bn1 = nn.BatchNorm1d(out_fea_len)
        self.act = nn.Softplus()     
        
    def forward(self, fea):
        fea = self.lin1(fea)
        fea = self.bn1(fea)
        fea = self.act(fea)

        return fea

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')# "Add" aggregation (Step 5).'add'/'mean','max'
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCNN(nn.Module):
    def __init__(self, orig_atom_fea_len, atom_fea_len):
        super(GCNN, self).__init__()
        self.conv1 = GCNConv(orig_atom_fea_len, atom_fea_len)
        self.conv2 = GCNConv(atom_fea_len, atom_fea_len)
        
        self.linear1 = nn.Linear(atom_fea_len, atom_fea_len)
        
    def forward(self, x):
        retval = []
        for _ in x:
            xx = self.conv1(_['node'], _['edge_idx'])
            retval.append(xx)
        
        return retval
        
class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
    ):
        """
        """
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)
        # self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
                )
                for _ in range(cry_heads)
            ]
        )

        # self.cry_pool = nn.ModuleList(
        #     [
        #         MeanPooling()
        #     ]
        # )

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):
        """
        Forward pass
        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch
        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx
        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
                # attnhead(elem_fea, index=cry_elem_idx)
            )

        return torch.mean(torch.stack(head_fea), dim=0)

    def __repr__(self):
        return self.__class__.__name__

class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        """
        """
        super().__init__()

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg),
                )
                for _ in range(elem_heads)
            ]
        )

        # self.pooling = nn.ModuleList(
        #     [
        #         MeanPooling()
        #     ]
        # )
        # self.mean_msg = SimpleNetwork(2*elem_fea_len, elem_fea_len, elem_msg)

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass
        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch
        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
                # attnhead(self.mean_msg(fea), index=self_fea_idx)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea
        # return fea

    def __repr__(self):
        return self.__class__.__name__


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn((1)))

    def forward(self, x, index, weights):
        """ forward pass """

        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self, input_dim, output_dim, hidden_layer_dims, activation=nn.LeakyReLU,
        batchnorm=False
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
                                    for i in range(len(dims)-1)])
        else:
            self.bns = nn.ModuleList([nn.Identity()
                                    for i in range(len(dims)-1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims, activation=nn.ReLU, batchnorm=False):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
                                    for i in range(len(dims)-1)])
        else:
            self.bns = nn.ModuleList([nn.Identity()
                                    for i in range(len(dims)-1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns,
                                       self.res_fcs, self.acts):
            x = act(bn(fc(x)))+res_fc(x)

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__


class Roost(nn.Module):
    """
    The Roost model is comprised of a fully connected network
    and message passing graph layers.
    The message passing layers are used to determine a descriptor set
    for the fully connected network. The graphs are used to represent
    the stoichiometry of inorganic materials in a trainable manner.
    This makes them systematically improvable with more data.
    """

    def __init__(
        self,
        task,
        robust,
        n_targets,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        out_hidden=[1024, 512, 256, 128, 64],
        #**kwargs
    ):
        super().__init__()

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
        }

        self.material_nn1 = DescriptorNetwork(**desc_dict)
        self.material_nn2 = DescriptorNetwork(**desc_dict)
        self.robust = robust

        # define an output neural network
        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets

        # self.output_nn = nn.Linear(elem_fea_len, output_dim)
        self.output_nn = ResidualNetwork(elem_fea_len*2, output_dim, out_hidden)
        # self.output_nn = SimpleNetwork(elem_fea_len, output_dim, out_hidden, nn.ReLU)

    def forward(self, input_tar, input_pre):
        """
        Forward pass through the material_nn and output_nn
        
        *input_tar
        ------------
        [0] : elem_weights
        [1] : elem_fea
        [2] : self_fea_idx
        [3] : nbr_fea_idx
        [4] : cry_elem_idx
        
        *input_pre
        ------------
        [0] : elem_weights
        [1] : elem_fea
        [2] : self_fea_idx
        [3] : nbr_fea_idx
        [4] : cry_elem_idx
        
        """
        tar_crys_fea = self.material_nn1(*input_tar)
        pre_crys_fea = self.material_nn2(*input_pre)
        syn_descriptor = torch.cat([tar_crys_fea, pre_crys_fea], dim=1)

        # apply neural network to map from learned features to target
        return self.output_nn(syn_descriptor), syn_descriptor
        #return self.output_nn(tar_crys_fea)

    def __repr__(self):
        return self.__class__.__name__

def RobustL1Loss(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * torch.exp(-log_std) + log_std
    return torch.mean(loss)

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, log=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.tensor((0))
        self.std = torch.tensor((1))

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key),\
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"],\
                                   layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn


def collate_batch(dataset_list):
    # define the lists
    batch_tar_atom_weights = []
    batch_tar_atom_fea = []
    batch_tar_self_fea_idx = []
    batch_tar_nbr_fea_idx = []
    batch_tar_atom_cry_idx = []
    batch_tar_metal_mask = []
    batch_tar_source_elem_idx = []
    
    batch_y = []
    batch_y2 = []
    batch_comp = []
    batch_ratio = []
    batch_cry_ids = []

    tar_cry_base_idx = 0
    source_n_i = 0
    for i, (x_tar_set, source_elem_idx, y, y2, y_stoi, y_ratio, cry_id) in enumerate(dataset_list):
        n_i = 0
        tar_set = []
        for j, x_tar in enumerate(x_tar_set):
            atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = x_tar[0]
            # number of atoms for this crystal
            n_ij = atom_fea.shape[0]

            # batch the features together
            batch_tar_atom_weights.append(atom_weights)
            batch_tar_atom_fea.append(atom_fea)

            # mappings from bonds to atoms
            batch_tar_self_fea_idx.append(self_fea_idx + tar_cry_base_idx)
            batch_tar_nbr_fea_idx.append(nbr_fea_idx + tar_cry_base_idx)

            # mapping metal_elem mask
            batch_tar_metal_mask.append(x_tar[2])
            
            # increment the id counter
            tar_cry_base_idx += n_ij
            n_i += n_ij
            tar_set.append(x_tar[1])
        
        # mapping from duplicated atoms to non-duplicated atoms
        batch_tar_source_elem_idx.append(torch.tensor(source_elem_idx)+source_n_i)
        source_n_i += max(source_elem_idx)+1
        
        # mapping from atoms to crystals
        batch_tar_atom_cry_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_y.append(y)
        batch_y2.append(y2)
        batch_comp.append((tar_set, y_stoi))
        batch_ratio += y_ratio
        batch_cry_ids.append(cry_id) 

    return (
        (
            torch.cat(batch_tar_atom_weights, dim=0),
            torch.cat(batch_tar_atom_fea, dim=0),
            torch.cat(batch_tar_self_fea_idx, dim=0),
            torch.cat(batch_tar_nbr_fea_idx, dim=0),
            torch.cat(batch_tar_atom_cry_idx),
        ),
        torch.cat(batch_tar_metal_mask, dim=0),
        torch.cat(batch_tar_source_elem_idx, dim=0),
        torch.cat(batch_y, dim=0),
        torch.cat(batch_y2, dim=0),
        batch_comp,
        batch_ratio,
        batch_cry_ids,
    )

def collate_batch2(dataset_list):
    # define the lists
    batch_tar_atom_weights = []
    batch_tar_atom_fea = []
    batch_tar_self_fea_idx = []
    batch_tar_nbr_fea_idx = []
    batch_tar_atom_cry_idx = []
    batch_tar_metal_mask = []
    batch_tar_source_elem_idx = []
    
    batch_pre_atom_weights = []
    batch_pre_atom_fea = []
    batch_pre_self_fea_idx = []
    batch_pre_nbr_fea_idx = []
    batch_pre_atom_cry_idx = []
    batch_pre_metal_mask = []
    batch_pre_source_elem_idx = []
    
    batch_y = []
    batch_comp = []
    batch_cry_ids = []

    tar_cry_base_idx = 0
    pre_cry_base_idx = 0
    tar_source_n_i = 0
    pre_source_n_i = 0
    for i, (x_tar_set, x_pre_set, tar_source_elem_idx, pre_source_elem_idx, y, cry_id) in enumerate(dataset_list):
        n_i = 0
        tar_set = []
        for j, x_tar in enumerate(x_tar_set):
            atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = x_tar[0]
            # number of atoms for this crystal
            n_ij = atom_fea.shape[0]

            # batch the features together
            batch_tar_atom_weights.append(atom_weights)
            batch_tar_atom_fea.append(atom_fea)

            # mappings from bonds to atoms
            batch_tar_self_fea_idx.append(self_fea_idx + tar_cry_base_idx)
            batch_tar_nbr_fea_idx.append(nbr_fea_idx + tar_cry_base_idx)

            # mapping metal_elem mask
            batch_tar_metal_mask.append(x_tar[2])
            
            # increment the id counter
            tar_cry_base_idx += n_ij
            n_i += n_ij
            tar_set.append(x_tar[1])
        
        # mapping from duplicated atoms to non-duplicated atoms
        batch_tar_source_elem_idx.append(torch.tensor(tar_source_elem_idx)+tar_source_n_i)
        tar_source_n_i += max(tar_source_elem_idx)+1
        
        # mapping from atoms to crystals
        batch_tar_atom_cry_idx.append(torch.tensor([i] * n_i))
        
        n_i = 0
        pre_set = []
        for j, x_pre in enumerate(x_pre_set):
            atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = x_pre[0]
            # number of atoms for this crystal
            n_ij = atom_fea.shape[0]

            # batch the features together
            batch_pre_atom_weights.append(atom_weights)
            batch_pre_atom_fea.append(atom_fea)

            # mappings from bonds to atoms
            batch_pre_self_fea_idx.append(self_fea_idx + pre_cry_base_idx)
            batch_pre_nbr_fea_idx.append(nbr_fea_idx + pre_cry_base_idx)

            # mapping metal_elem mask
            batch_pre_metal_mask.append(x_pre[2])
            
            # increment the id counter
            pre_cry_base_idx += n_ij
            n_i += n_ij
            pre_set.append(x_pre[1])
        
        # mapping from duplicated atoms to non-duplicated atoms
        batch_pre_source_elem_idx.append(torch.tensor(pre_source_elem_idx)+pre_source_n_i)
        pre_source_n_i += max(pre_source_elem_idx)+1
        
        # mapping from atoms to crystals
        batch_pre_atom_cry_idx.append(torch.tensor([i] * n_i))
        
        # batch the targets and ids
        batch_y.append(y)
        batch_comp.append((tar_set, pre_set))
        batch_cry_ids.append(cry_id)
        
    
    return (
        (
            torch.cat(batch_tar_atom_weights, dim=0),
            torch.cat(batch_tar_atom_fea, dim=0),
            torch.cat(batch_tar_self_fea_idx, dim=0),
            torch.cat(batch_tar_nbr_fea_idx, dim=0),
            torch.cat(batch_tar_atom_cry_idx),
            torch.cat(batch_tar_metal_mask, dim=0),
            torch.cat(batch_tar_source_elem_idx, dim=0),
        ),
        (
            torch.cat(batch_pre_atom_weights, dim=0),
            torch.cat(batch_pre_atom_fea, dim=0),
            torch.cat(batch_pre_self_fea_idx, dim=0),
            torch.cat(batch_pre_nbr_fea_idx, dim=0),
            torch.cat(batch_pre_atom_cry_idx),
            torch.cat(batch_pre_metal_mask, dim=0),
            torch.cat(batch_pre_source_elem_idx, dim=0),
        ),
        torch.stack(batch_y, dim=0).reshape(-1,1),
        batch_comp,
        batch_cry_ids,
    )

