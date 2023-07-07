import json
import copy
import random
import sys
import numpy as np
import pickle as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_min

from pymatgen.core import Composition
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from Data import get_SourceElem, get_AnionPart, get_Source_Anion_ratio
from Model import PrecursorClassifier, collate_batch

random.seed(8888)
torch.manual_seed(8888)
np.random.seed(8888)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def composition2graph(composition, embedding_dict):
    comp_dict = Composition(composition).get_el_amt_dict()
    elements_seq = list(comp_dict.keys())
    weights = list(comp_dict.values())
    weights = np.atleast_2d(weights).T / np.sum(weights)

    try:
        atom_fea = np.vstack(
            [np.array(embedding_dict[str(element)]) for element in elements_seq]
        )
    except Exception as ex:
        raise NotImplementedError(
            f"{ex} in '{composition}' has no embedding vector"
        )

    env_idx = list(range(len(elements_seq)))
    if len(env_idx) == 1:
        self_fea_idx = [0]
        nbr_fea_idx = [0]
    else:
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elements_seq) - 1
        for i, _ in enumerate(elements_seq):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

    # convert all data to tensors
    atom_weights = torch.Tensor(weights)
    atom_fea = torch.Tensor(atom_fea)
    self_fea_idx = torch.LongTensor(self_fea_idx)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

    return ((atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            composition,
            ), elements_seq

def composition2fea(composition, embedding_dict):
    comp_dict = Composition(composition).get_el_amt_dict()
    elements_seq = list(comp_dict.keys())
    weights = list(comp_dict.values())
    weights = np.atleast_2d(weights).T / np.sum(weights)

    try:
        atom_fea = np.vstack(
            [np.array(embedding_dict[str(element)]) for element in elements_seq]
        )
    except Exception as ex:
        raise NotImplementedError(
            f"{ex} in '{composition}' has no embedding vector"
        )
    comp_fea = []
    for i in range(len(atom_fea)):
        comp_fea.append(atom_fea[i] * weights[i])
    comp_fea = np.array(comp_fea)
    comp_fea = comp_fea.sum(axis=0)

    return comp_fea

def add_source_mask(graph, source_elem):
    composition = graph[1]
    comp_dict = Composition(composition).get_el_amt_dict()
    mask_vec = []
    for elem, stoi in comp_dict.items():
        if str(elem) in source_elem:
            mask_vec.append([source_elem.index(elem)])
        else:
            mask_vec.append([-1])
    return (graph[0], graph[1], torch.tensor(mask_vec))

def anion_labeling(composition, pre_anion_part, source_elem):
    pre_anion_part = list(pre_anion_part)
    class_len = len(pre_anion_part)
    y_label = np.zeros(class_len)

    anion = get_AnionPart(composition, source_elem)

    for i in range(class_len):
        if anion == pre_anion_part[i]:
            y_label[i] = 1
    if sum(y_label) != 1:
        raise NotImplementedError('labeling error')

    return y_label

# Check model
def find_precursors_indiv(tar_composition,
                          top_k,
                          model,
                          device,embedding_dict,pre_anion_part,stoi_dict):

    dataset = []
    x_tar_set = []
    elements_seq_set = []
    for j in range(len(tar_composition)):
        x_tar, elements_seq = composition2graph(tar_composition[j], embedding_dict)
        x_tar = add_source_mask(x_tar, get_SourceElem([tar_composition[j]])[0])
        x_tar_set.append(x_tar)
        elements_seq_set.append(elements_seq)
    source_elem_seq = []
    source_elem_idx = []
    count=0
    for elem_seq in elements_seq_set:
        for elem in elem_seq:
            if (elem in get_SourceElem(tar_composition)[0]) and (elem not in source_elem_seq):
                source_elem_seq.append(elem)
                source_elem_idx.append(count)
                count+=1
            elif (elem in get_SourceElem(tar_composition)[0]) and (elem in source_elem_seq):
                source_elem_idx.append(source_elem_seq.index(elem))

    y = []
    y2 = []
    y_stoi = []
    y_ratio = []
    y.append(np.zeros(len(pre_anion_part)))
    y2.append(np.zeros(len(embedding_dict['Li'])))
    y_stoi.append('')
    y_ratio.append([])
    y = torch.Tensor(np.array(y))
    y2 = torch.Tensor(np.array(y2))
    dataset.append((x_tar_set, source_elem_idx, y, y2, y_stoi, y_ratio, 0))

    input_tar, metal_mask, source_elem_idx, batch_y, batch_y2, batch_comp, batch_ratio, batch_i = collate_batch(dataset)
    input_tar = tuple([tensor.to(device) for tensor in input_tar])
    metal_mask = metal_mask.to(device)
    source_elem_idx = source_elem_idx.to(device)
    batch_y = batch_y.to(device)
    batch_y = torch.where(batch_y==1)[1]
    pre_set_idx = scatter_mean(input_tar[4][torch.where(metal_mask!=-1)[0]], source_elem_idx, dim=0)

    # compute output
    template_output, sourceelem_descriptor = model(input_tar, metal_mask, source_elem_idx, pre_set_idx)

    score_matrix = []
    pred_matrix = []
    for k in range(top_k):
        score_matrix.append(torch.kthvalue(F.softmax(template_output, dim=1), template_output.shape[1]-k)[0])
        pred_matrix.append(torch.kthvalue(F.softmax(template_output, dim=1), template_output.shape[1]-k)[1])
    score_matrix = torch.stack(score_matrix, dim=0)
    pred_matrix = torch.stack(pred_matrix, dim=0)

    kth_precursors = []
    for k in range(top_k):
        precursors_set = []
        for l in range(len(source_elem_seq)):
            pre_score = round(score_matrix[k,l].item(), 4)
            source_part = source_elem_seq[l]
            counter_part = list(pre_anion_part)[pred_matrix[k][l]]
            stoi_space = stoi_dict[source_part+counter_part]
            if len(stoi_space) == 0:
                precursors_set.append(('('+source_part+')('+counter_part+')', pre_score))
            else:
                precursor = stoi_space[0]
                precursors_set.append((precursor, pre_score))

        kth_precursors.append(precursors_set)

    return kth_precursors

def find_precursors_set(tar_composition,
                        top_k,
                        model,
                        device,embedding_dict,pre_anion_part,stoi_dict):

    dataset = []
    x_tar_set = []
    elements_seq_set = []
    for j in range(len(tar_composition)):
        x_tar, elements_seq = composition2graph(tar_composition[j], embedding_dict)
        x_tar = add_source_mask(x_tar, get_SourceElem([tar_composition[j]])[0])
        x_tar_set.append(x_tar)
        elements_seq_set.append(elements_seq)
    source_elem_seq = []
    source_elem_idx = []
    count=0
    for elem_seq in elements_seq_set:
        for elem in elem_seq:
            if (elem in get_SourceElem(tar_composition)[0]) and (elem not in source_elem_seq):
                source_elem_seq.append(elem)
                source_elem_idx.append(count)
                count+=1
            elif (elem in get_SourceElem(tar_composition)[0]) and (elem in source_elem_seq):
                source_elem_idx.append(source_elem_seq.index(elem))

    y = []
    y2 = []
    y_stoi = []
    y_ratio = []
    y.append(np.zeros(len(pre_anion_part)))
    y2.append(np.zeros(len(embedding_dict['Li'])))
    y_stoi.append('')
    y_ratio.append([])
    y = torch.Tensor(np.array(y))
    y2 = torch.Tensor(np.array(y2))
    dataset.append((x_tar_set, source_elem_idx, y, y2, y_stoi, y_ratio, 0))

    input_tar, metal_mask, source_elem_idx, batch_y, batch_y2, batch_comp, batch_ratio, batch_i = collate_batch(dataset)
    input_tar = tuple([tensor.to(device) for tensor in input_tar])
    metal_mask = metal_mask.to(device)
    source_elem_idx = source_elem_idx.to(device)
    batch_y = batch_y.to(device)
    batch_y = torch.where(batch_y==1)[1]
    pre_set_idx = scatter_mean(input_tar[4][torch.where(metal_mask!=-1)[0]], source_elem_idx, dim=0)

    # compute output
    template_output, sourceelem_descriptor = model(input_tar, metal_mask, source_elem_idx, pre_set_idx)

    score_matrix = []
    pred_matrix = []
    for k in range(top_k):
        score_matrix.append(torch.kthvalue(F.softmax(template_output, dim=1), template_output.shape[1]-k)[0])
        pred_matrix.append(torch.kthvalue(F.softmax(template_output, dim=1), template_output.shape[1]-k)[1])
    score_matrix = torch.stack(score_matrix, dim=0)
    pred_matrix = torch.stack(pred_matrix, dim=0)

    set_score_list = []
    set_num = template_output.shape[0]

    for elem_idx in range(set_num):
        if elem_idx == 0:
            set_score_list = score_matrix[:,elem_idx:elem_idx+1]
        else:
            set_score_list = torch.matmul(set_score_list, score_matrix[:,elem_idx:elem_idx+1].T).reshape(-1,1)

    top_k_result = []
    for k in range(top_k):
        kst_score = round(torch.kthvalue(set_score_list.T, len(set_score_list)-k)[0].item(), 4)
        kst_idx = torch.kthvalue(set_score_list.T, len(set_score_list)-k)[1].item()
        kst_pre_set = []
        for idx in range(set_num):
            kst_pre_set.append(pred_matrix[int(kst_idx/(top_k**(set_num-idx-1))), idx].item())
            kst_idx = kst_idx % (top_k**(set_num-idx-1))
        top_k_result.append((kst_pre_set, kst_score))

    kth_precursors = []
    for k in range(len(top_k_result)):
        set_score = top_k_result[k][1]
        precursors_set = []
        for l in range(len(source_elem_seq)):
            source_part = source_elem_seq[l]
            counter_part = list(pre_anion_part)[top_k_result[k][0][l]]
            stoi_space = stoi_dict[source_part+counter_part]
            if len(stoi_space) == 0:
                precursors_set.append('('+source_part+')('+counter_part+')')
            else:
                precursor = stoi_space[0]
                precursors_set.append(precursor)

        kth_precursors.append((precursors_set, set_score))

    return kth_precursors


if __name__ == "__main__":

    with open("embedding/cgcnn-embedding.json", 'r', encoding='utf-8-sig') as json_file:
        cgcnn = json.load(json_file)
    with open("embedding/elem-embedding.json", 'r', encoding='utf-8-sig') as json_file:
        elemnet = json.load(json_file)
    with open("embedding/matscholar-embedding.json", 'r', encoding='utf-8-sig') as json_file:
        matscholar = json.load(json_file)
    with open("embedding/megnet16-embedding.json", 'r', encoding='utf-8-sig') as json_file:
        megnet16 = json.load(json_file)
    with open("embedding/onehot-embedding.json", 'r', encoding='utf-8-sig') as json_file:
        onehot = json.load(json_file)
    with open("embedding/cgcnn_hd_rcut4_nn8.element_embedding.json", 'r', encoding='utf-8-sig') as json_file:
        cgcnn_hd = json.load(json_file)

    embedding_dict = elemnet


    training_type = input("Which model to test ? (RandSplit or TimeSplit) = ")
    if training_type not in ['RandSplit', 'TimeSplit']:
        print("Invalid name of model type !")
        sys.exit()
    if training_type == 'RandSplit':
        check_time_transferability = False
    else:
        check_time_transferability = True

    # Prepare data
    if check_time_transferability == False:
        file_path = "./dataset/InorgSyn_dataset_TP.json"
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    else:
        print("-----------Checking time transferability-----------")
        file_path = "./dataset/InorgSyn_dataset_TP2.json"
        with open(file_path, "r") as json_file:
            data2 = json.load(json_file)

        data_in = []
        data_out = []
        from datetime import datetime
        for dd in data2:
            if dd['pubdate'] != 'N/A':
                pub_date1 = datetime.strptime(dd['pubdate'], "%Y-%m-%d")
                pub_date2 = datetime.strptime("2016-01-01", "%Y-%m-%d")
                day_diff = (pub_date2 - pub_date1).days
                if day_diff <= 0:
                    data_out.append(dd)
                else:
                    data_in.append(dd)

        data = data_in + data_out
        print(len(data_in), len(data_out), len(data))

    file_path = "./dataset/pre_anion_part.json"
    with open(file_path, "r") as json_file:
        pre_anion_part = json.load(json_file)

    file_path = "./dataset/stoi_dict.json"
    with open(file_path, "r") as json_file:
        stoi_dict = json.load(json_file)

    dataset = []
    for i in range(len(data)):
        x_tar_set = []
        elements_seq_set = []
        for j in range(len(data[i]['Target'])):
            x_tar, elements_seq = composition2graph(data[i]['Target'][j], embedding_dict)
            x_tar = add_source_mask(x_tar, get_SourceElem([data[i]['Target'][j]])[0])
            x_tar_set.append(x_tar)
            elements_seq_set.append(elements_seq)
        source_elem_seq = []
        source_elem_idx = []
        count=0
        for elem_seq in elements_seq_set:
            for elem in elem_seq:
                if (elem in get_SourceElem(data[i]['Target'])[0]) and (elem not in source_elem_seq):
                    source_elem_seq.append(elem)
                    source_elem_idx.append(count)
                    count+=1
                elif (elem in get_SourceElem(data[i]['Target'])[0]) and (elem in source_elem_seq):
                    source_elem_idx.append(source_elem_seq.index(elem))

        y = []
        y2 = []
        y_stoi = []
        y_ratio = []
        for elem in source_elem_seq:
            for j in range(len(data[i]['Precursors'])):
                if elem in list(Composition(data[i]['Precursors'][j]).get_el_amt_dict().keys()):
                    y.append(anion_labeling(data[i]['Precursors'][j], pre_anion_part, get_SourceElem(data[i]['Target'])[0]))
                    y2.append(composition2fea(data[i]['Precursors'][j], embedding_dict))
                    y_stoi.append(data[i]['Precursors'][j])
                    y_ratio.append(get_Source_Anion_ratio([data[i]['Precursors'][j]])[0])
        if max(source_elem_idx)+1 != len(y):
            raise NotImplementedError('labeling error')

        y = torch.Tensor(np.array(y))
        y2 = torch.Tensor(np.array(y2))
        dataset.append((x_tar_set, source_elem_idx, y, y2, y_stoi, y_ratio, i))

    def ratio2composition(s_elem, pre_template, r_ratio):
        if pre_template != '':
            ratio_comp = s_elem+'('+pre_template+')'+str(r_ratio)
            r_int_comp = Composition(ratio_comp).get_integer_formula_and_factor()[0]
        else:
            r_int_comp = s_elem
        return r_int_comp

    def check_same_composition(ratio_comp, original_comp):
        r_int_comp = Composition(ratio_comp).get_integer_formula_and_factor()[0]
        o_int_comp = Composition(original_comp).get_integer_formula_and_factor()[0]
        check = (r_int_comp == o_int_comp)
        return check

    for dd in dataset:
        pre_list = dd[4]
        r_stoi_list = dd[5]
        for j, pre in enumerate(pre_list):

            p_source_elem, _ = get_SourceElem([pre])
            template = get_AnionPart(pre, p_source_elem)
            if len(p_source_elem) == 1:
                r_int_comp = ratio2composition(p_source_elem[0], template, r_stoi_list[j])
            else:
                raise NotImplementedError("No single source element precursors")
            if check_same_composition(r_int_comp, pre):
                pass
            else:
                print(p_source_elem[0], template, r_stoi_list[j], r_int_comp, pre)

    if check_time_transferability == False:
        train_set, test_set = train_test_split(dataset, test_size=0.1, random_state=7)
        train_set, val_set = train_test_split(train_set, test_size=0.1112, random_state=7)
    else:
        train_set, test_set = dataset[:len(data_in)], dataset[len(data_in):]
        train_set, val_set = train_test_split(train_set, test_size=0.1112, random_state=7)

    if check_time_transferability == False:
        print("Total dataset size : %d, (train/val/test = %d/%d/%d = 8:1:1)" % (len(dataset), len(train_set), len(val_set), len(test_set)))
    else:
        print("Total dataset size : %d, (train/val/test = %d/%d/%d)" % (len(dataset), len(train_set), len(val_set), len(test_set)))

    data_params = {"batch_size": 128, "num_workers": 0, "pin_memory": False,
                   "shuffle": False, "collate_fn": collate_batch,
                   "worker_init_fn": seed_worker}

    train_idx = list(range(len(train_set)))
    val_idx = list(range(len(val_set)))
    test_idx = list(range(len(test_set)))

    train_set = torch.utils.data.Subset(train_set, train_idx[0::1])
    val_set = torch.utils.data.Subset(val_set, val_idx[0::1])
    test_set = torch.utils.data.Subset(test_set, test_idx[0::1])
    train_generator = DataLoader(train_set, **data_params)
    val_generator = DataLoader(val_set, **data_params)
    test_generator = DataLoader(test_set, **data_params)


    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    Available model_mode ;
        1. pooling_mode, globalfactor, gru_mode = False, False, False (ElemwiseRetro)
        2. pooling_mode, globalfactor, gru_mode = False, False, True (Source elem-wise)
        3. pooling_mode, globalfactor, gru_mode = False, True, True (Source elem-wise w. GLA)
        4. pooling_mode, globalfactor, gru_mode = True, False, True (GLA)
    """
    pooling_mode = False    # True : weighted-attentioned mean pooling / False : source element-wise
    globalfactor = False    # True : concatenate initial node vector with global pooling vector
    gru_mode = False        # True : After GCN, GRU prediction         / False : After GCN, ResNet prediction
    print('[Pooling', pooling_mode, ', Globalfactor', globalfactor, ', GRU', gru_mode, '] mode')

    model_params = {
            "task": "Classification",
            "pooling": pooling_mode,
            "globalfactor": globalfactor,
            "gru": gru_mode,
            "device": device,
            "robust": False,
            "n_targets": len(pre_anion_part),
            "elem_emb_len": len(embedding_dict['Li']),
            "elem_fea_len": 64,
            "n_graph": 3,
            "elem_heads": 3,
            "elem_gate": [256],
            "elem_msg": [256],
            "cry_heads": 3,
            "cry_gate": [256],
            "cry_msg": [256],
            #"out_hidden": [1024, 512, 256, 128, 64],
            "out_hidden": [512, 512, 512],
        }
    model =  PrecursorClassifier(**model_params)

    # Prepare learning parameters
    num_epoch = 50
    criterion = nn.CrossEntropyLoss()
    lr = 3e-4
    weight_decay=1e-6
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Train process
    model.to(device)

    train_loss_curve = []
    val_loss_curve = []

    best_val_loss = 10000000
    best_model_wts = copy.deepcopy(model.state_dict())
    for i in range(num_epoch):
        loss_list = []
        model.train()
        for input_tar, metal_mask, source_elem_idx, batch_y, batch_y2, batch_comp, batch_ratio, batch_i in train_generator:
            # move tensors to device (GPU or CPU)
            input_tar = tuple([tensor.to(device) for tensor in input_tar])
            metal_mask = metal_mask.to(device)
            source_elem_idx = source_elem_idx.to(device)
            batch_y = batch_y.to(device)
            batch_y_id = torch.where(batch_y==1)[1]
            batch_y2 = batch_y2.to(device)
            pre_set_idx = scatter_mean(input_tar[4][torch.where(metal_mask!=-1)[0]], source_elem_idx, dim=0)

            # compute output
            template_output, atomic_descriptor = model(input_tar, metal_mask, source_elem_idx, pre_set_idx)
            loss = criterion(template_output, batch_y_id)
            loss_list.append(loss.data.cpu().numpy())

            # compute gradient and take an optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(np.array(loss_list))

        val_loss_list = []
        model.eval()
        with torch.no_grad(): # Make zero gradient
            for input_tar, metal_mask, source_elem_idx, batch_y, batch_y2, batch_comp, batch_ratio, batch_i in val_generator:
                # move tensors to device (GPU or CPU)
                input_tar = tuple([tensor.to(device) for tensor in input_tar])
                metal_mask = metal_mask.to(device)
                source_elem_idx = source_elem_idx.to(device)
                batch_y = batch_y.to(device)
                batch_y_id = torch.where(batch_y==1)[1]
                batch_y2 = batch_y2.to(device)
                pre_set_idx = scatter_mean(input_tar[4][torch.where(metal_mask!=-1)[0]], source_elem_idx, dim=0)

                # compute output
                template_output, atomic_descriptor = model(input_tar, metal_mask, source_elem_idx, pre_set_idx)
                loss = criterion(template_output, batch_y_id)
                val_loss_list.append(loss.data.cpu().numpy())

        val_loss = np.mean(np.array(val_loss_list))
        if (i+1)%10==0:
            print ('Epoch ', i+1, ', training loss: ', train_loss, ', val loss: ',val_loss)
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts) # load best model weights

    model.eval()

    # Test
    def find_top_k_prediction(output, pre_set_idx, top_k):
        slicing_idx = []
        for i in range(len(pre_set_idx)):
            if i == 0:
                prev_num = pre_set_idx[i]
                slicing_idx.append(i)
            else:
                if prev_num == pre_set_idx[i]:
                    pass
                else:
                    prev_num = pre_set_idx[i]
                    slicing_idx.append(i)
        slicing_idx.append(len(pre_set_idx))

        kth_output = []
        for i, idx in enumerate(slicing_idx):
            if i == 0:
                prev_i = idx
            else:
                if i != len(slicing_idx)-1:
                    sliced_output = output[prev_i:idx]
                    prev_i = idx
                else:
                    sliced_output = output[prev_i:]

                score_matrix = []
                pred_matrix = []
                for k in range(top_k):
                    score_matrix.append(torch.kthvalue(F.softmax(sliced_output, dim=1), sliced_output.shape[1]-k)[0])
                    pred_matrix.append(torch.kthvalue(F.softmax(sliced_output, dim=1), sliced_output.shape[1]-k)[1])
                score_matrix = torch.stack(score_matrix, dim=0)
                pred_matrix = torch.stack(pred_matrix, dim=0)

                set_score_list = []
                set_num = sliced_output.shape[0]

                for elem_idx in range(set_num):
                    if elem_idx == 0:
                        set_score_list = score_matrix[:,elem_idx:elem_idx+1]
                    else:
                        set_score_list = torch.matmul(set_score_list, score_matrix[:,elem_idx:elem_idx+1].T).reshape(-1,1)

                sliced_top_k_result = []
                for k in range(top_k):
                    #kst_score = round(torch.kthvalue(set_score_list.T, len(set_score_list)-k)[0].item(), 4)
                    kst_idx = torch.kthvalue(set_score_list.T, len(set_score_list)-k)[1].item()
                    kst_pre_set = []
                    for idx in range(set_num):
                        kst_pre_set.append(pred_matrix[int(kst_idx/(top_k**(set_num-idx-1))), idx].item())
                        kst_idx = kst_idx % (top_k**(set_num-idx-1))
                    #sliced_top_k_result.append((kst_pre_set, kst_score))
                    sliced_top_k_result.append(kst_pre_set)

                kth_output.append(torch.tensor(sliced_top_k_result))
        return torch.cat(kth_output, dim=1)

    model.eval()

    pred_value_te = []
    true_value_te = []
    pre_set_idx_te = []
    top_k_pred_te = []
    idx_te = []

    total_batch_precursors = []
    total_kth_pred_precursors = {}
    for k in range(5):
        total_kth_pred_precursors['Top-'+str(k+1)] = []

    total_pre_set_count_te = 0
    with torch.no_grad(): # Make zero gradient
        for input_tar, metal_mask, source_elem_idx, batch_y, batch_y2, batch_comp, batch_ratio, batch_i in test_generator:
            # move tensors to device (GPU or CPU)
            input_tar = tuple([tensor.to(device) for tensor in input_tar])
            metal_mask = metal_mask.to(device)
            source_elem_idx = source_elem_idx.to(device)
            batch_y = batch_y.to(device)
            batch_y = torch.where(batch_y==1)[1]
            pre_set_idx = scatter_mean(input_tar[4][torch.where(metal_mask!=-1)[0]], source_elem_idx, dim=0)
            batch_targets = []
            batch_precursors = []
            for i in range(len(batch_comp)):
                batch_targets.append(batch_comp[i][0])
                for j in range(len(batch_comp[i][1])):
                    batch_precursors.append(batch_comp[i][1][j])

            # compute output
            template_output, atomic_descriptor = model(input_tar, metal_mask, source_elem_idx, pre_set_idx)
            pred = template_output.max(dim=1)[1]

            true = batch_y
            pred_value_te += pred.tolist()
            true_value_te += true.tolist()
            idx_te += batch_i

            # compute precursors set index
            pre_set_idx += total_pre_set_count_te
            pre_set_idx_te.append(pre_set_idx)
            total_pre_set_count_te += len(batch_comp)
            total_batch_precursors += batch_precursors

            # compute top-k precursors set index
            top_k_pred = find_top_k_prediction(template_output.cpu(), pre_set_idx.cpu(), 5)
            top_k_pred_te.append(top_k_pred)

    template_pred_value_te = np.array(pred_value_te)
    template_true_value_te = np.array(true_value_te)
    pre_set_idx_te= torch.cat(pre_set_idx_te, dim=0)
    template_top_k_pred_te = torch.cat(top_k_pred_te, dim=1)
    idx_te = np.array(idx_te)

    accuracy_result = {}

    template_te_accuracy = sum(template_pred_value_te == template_true_value_te)/len(template_true_value_te)
    print("Accuracy for individual_precursor_template of testset :", round(template_te_accuracy,4))
    accuracy_result['indiv_template_acc'] = round(template_te_accuracy,4)
    for k in range(template_top_k_pred_te.shape[0]):
        if k == 0:
            b = torch.tensor((template_top_k_pred_te[k].numpy() == template_true_value_te), dtype=float)
            te_set_accuracy = sum(scatter_min(torch.tensor((b>0).numpy(), dtype=float), pre_set_idx_te.cpu(), dim=0)[0])/len(test_set)
            print("Top-%d Accuracy for precursors_template_set of testset : %f" %(k+1, round(float(te_set_accuracy),4)))
            accuracy_result['Top-'+str(k+1)+'_template_set_acc'] = round(float(te_set_accuracy),4)
        else:
            b = b + torch.tensor((template_top_k_pred_te[k].numpy() == template_true_value_te), dtype=float)
            te_set_accuracy = sum(scatter_min(torch.tensor((b>0).numpy(), dtype=float), pre_set_idx_te.cpu(), dim=0)[0])/len(test_set)
            print("Top-%d Accuracy for precursors_template_set of testset : %f" %(k+1, round(float(te_set_accuracy),4)))
            accuracy_result['Top-'+str(k+1)+'_template_set_acc'] = round(float(te_set_accuracy),4)
      
    train_val_loss = {'Model_train_loss_curve' : train_loss_curve,
                      'Model_val_loss_curve'   : val_loss_curve,
                      }
    
    if check_time_transferability == False:
        pk.dump(idx_te, open('./dataset/test_idx_TP.sav', 'wb'))
        pk.dump(dataset, open('./dataset/preprocessed_data_TP.sav', 'wb'))
        pk.dump(model, open('./model/trained_model_TP_'+str(pooling_mode)+str(globalfactor)+str(gru_mode)+'.sav', 'wb'))
        pk.dump(accuracy_result, open('./result/accuracy_result_TP_'+str(pooling_mode)+str(globalfactor)+str(gru_mode)+'.sav', 'wb'))
        pk.dump(train_val_loss, open('./result/train_val_loss_TP_'+str(pooling_mode)+str(globalfactor)+str(gru_mode)+'.sav', 'wb'))
    else:
        pk.dump(idx_te, open('./dataset/test_idx_TP_time.sav', 'wb'))
        pk.dump(dataset, open('./dataset/preprocessed_data_TP_time.sav', 'wb'))
        pk.dump(model, open('./model/trained_model_TP_'+str(pooling_mode)+str(globalfactor)+str(gru_mode)+'_time.sav', 'wb'))
        pk.dump(accuracy_result, open('./result/accuracy_result_TP_'+str(pooling_mode)+str(globalfactor)+str(gru_mode)+'_time.sav', 'wb'))
        pk.dump(train_val_loss, open('./result/train_val_loss_TP_'+str(pooling_mode)+str(globalfactor)+str(gru_mode)+'_time.sav', 'wb'))
