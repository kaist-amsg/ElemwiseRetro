import json
import copy
import random
import numpy as np
import pickle as pk
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from Data import get_SourceElem, get_AnionPart, get_Source_Anion_ratio
from Model import TemperatureRegressor, RobustL1Loss, Normalizer, collate_batch2
from Train_P import seed_worker, composition2graph, add_source_mask

random.seed(8888)
torch.manual_seed(8888)
np.random.seed(8888)


def find_syn_temperature(tar_composition, pre_composition,
                         model_T,normalizer,device,
                         embedding_dict):
    dataset = []
    
    x_tar_set = []
    elements_seq_set = []
    for j in range(len(tar_composition)):
        x_tar, elements_seq = composition2graph(tar_composition[j], embedding_dict)
        x_tar = add_source_mask(x_tar, get_SourceElem([tar_composition[j]])[0])
        x_tar_set.append(x_tar)
        elements_seq_set.append(elements_seq)
    source_elem_seq = []
    tar_source_elem_idx = []
    count=0
    for elem_seq in elements_seq_set:
        for elem in elem_seq:
            if (elem in get_SourceElem(tar_composition)[0]) and (elem not in source_elem_seq):
                source_elem_seq.append(elem)
                tar_source_elem_idx.append(count)
                count+=1
            elif (elem in get_SourceElem(tar_composition)[0]) and (elem in source_elem_seq):
                tar_source_elem_idx.append(source_elem_seq.index(elem))
    
    x_pre_set = []
    elements_seq_set = []
    for j in range(len(pre_composition)):
        x_pre, elements_seq = composition2graph(pre_composition[j], embedding_dict)
        x_pre = add_source_mask(x_pre, get_SourceElem([pre_composition[j]])[0])
        x_pre_set.append(x_pre)
        elements_seq_set.append(elements_seq)
    source_elem_seq = []
    pre_source_elem_idx = []
    count=0
    for elem_seq in elements_seq_set:
        for elem in elem_seq:
            if (elem in get_SourceElem([pre_composition[j]])[0]) and (elem not in source_elem_seq):
                source_elem_seq.append(elem)
                pre_source_elem_idx.append(count)
                count+=1
            elif (elem in get_SourceElem([pre_composition[j]])[0]) and (elem in source_elem_seq):
                pre_source_elem_idx.append(source_elem_seq.index(elem))
    if max(tar_source_elem_idx) != max(pre_source_elem_idx):
        raise NotImplementedError('labeling error')
    
    y_set = [0.0]
    y = torch.mean(torch.tensor(y_set))
    dataset.append((x_tar_set, x_pre_set, tar_source_elem_idx, pre_source_elem_idx, y, 0))

    input_tar, input_pre, batch_y, batch_comp, batch_i = collate_batch2(dataset)
    
    # move tensors to device (GPU or CPU)
    input_tar = tuple([tensor.to(device) for tensor in input_tar])
    input_pre = tuple([tensor.to(device) for tensor in input_pre])
    
    # compute output
    output, syn_descriptor = model_T(input_tar, input_pre)
    output, log_std = output.chunk(2, dim=1)
    pred = normalizer.denorm(output.data.cpu())
    
    return round(pred[0].item(), 1)


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
    temp_min = 300
    temp_max = 1600
    
    
    # Prepare data
    file_path = "./dataset/InorgSyn_dataset_TPO.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    
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
        tar_source_elem_idx = []
        count=0
        for elem_seq in elements_seq_set:
            for elem in elem_seq:
                if (elem in get_SourceElem(data[i]['Target'])[0]) and (elem not in source_elem_seq):
                    source_elem_seq.append(elem)
                    tar_source_elem_idx.append(count)
                    count+=1
                elif (elem in get_SourceElem(data[i]['Target'])[0]) and (elem in source_elem_seq):
                    tar_source_elem_idx.append(source_elem_seq.index(elem))
        
        x_pre_set = []
        elements_seq_set = []
        for j in range(len(data[i]['Precursors'])):
            x_pre, elements_seq = composition2graph(data[i]['Precursors'][j], embedding_dict)
            x_pre = add_source_mask(x_pre, get_SourceElem([data[i]['Precursors'][j]])[0])
            x_pre_set.append(x_pre)
            elements_seq_set.append(elements_seq)
        source_elem_seq = []
        pre_source_elem_idx = []
        count=0
        for elem_seq in elements_seq_set:
            for elem in elem_seq:
                if (elem in get_SourceElem(data[i]['Precursors'])[0]) and (elem not in source_elem_seq):
                    source_elem_seq.append(elem)
                    pre_source_elem_idx.append(count)
                    count+=1
                elif (elem in get_SourceElem(data[i]['Precursors'])[0]) and (elem in source_elem_seq):
                    pre_source_elem_idx.append(source_elem_seq.index(elem))
        if max(tar_source_elem_idx) != max(pre_source_elem_idx):
            raise NotImplementedError('labeling error')
        
        y_set = []
        for j in range(len(data[i]['Operation'])):
            y = float(data[i]['Operation'][j])
            y_set.append(y)
        y = torch.mean(torch.tensor(y_set))
        
        dataset.append((x_tar_set, x_pre_set, tar_source_elem_idx, pre_source_elem_idx, y, i))
    
    train_set, test_set = train_test_split(dataset, test_size=0.1, random_state=774)
    train_set, val_set = train_test_split(train_set, test_size=0.1112, random_state=774)
    
    print("Total dataset size : %d, (train/val/test = %d/%d/%d = 8:1:1)" % (len(dataset), len(train_set), len(val_set), len(test_set)))
    
    data_params = {"batch_size": 128, "num_workers": 0, "pin_memory": False,
                   "shuffle": False, "collate_fn": collate_batch2,
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
    
    sample_y = torch.tensor([[y_value] for _, _, _, _, y_value, _ in train_set])
    normalizer = Normalizer()
    normalizer.fit(sample_y)
    
    
    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_params = {
            "task": "Regression",
            "pooling": True,
            "globalfactor": False,
            "device": device,
            "robust": True,
            "n_targets": 1,
            "elem_emb_len": len(embedding_dict['Li']),
            "elem_fea_len": 64,
            "n_graph": 3,
            "elem_heads": 3,
            "elem_gate": [256],
            "elem_msg": [256],
            "cry_heads": 3,
            "cry_gate": [256],
            "cry_msg": [256],
            "out_hidden": [1024, 512, 256, 128, 64],
            #"out_hidden": [512, 512, 512],
        }
    model =  TemperatureRegressor(**model_params)
    
    
    # Prepare learning parameters
    num_epoch = 50
    criterion = RobustL1Loss
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
        for input_tar, input_pre, batch_y, batch_comp, batch_i in train_generator:
            # move tensors to device (GPU or CPU)
            input_tar = tuple([tensor.to(device) for tensor in input_tar])
            input_pre = tuple([tensor.to(device) for tensor in input_pre])
            batch_y = normalizer.norm(batch_y)
            batch_y = batch_y.to(device)
            
            # compute output
            output, syn_descriptor = model(input_tar, input_pre)
            output, log_std = output.chunk(2, dim=1)
            loss = criterion(output, log_std, batch_y)
            loss_list.append(loss.data.cpu().numpy())
            #pred = normalizer.denorm(output.data.cpu())
            
            # compute gradient and take an optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = np.mean(np.array(loss_list))
    
        val_loss_list = []
        model.eval()
        with torch.no_grad(): # Make zero gradient
            for input_tar, input_pre, batch_y, batch_comp, batch_i in val_generator:
                # move tensors to device (GPU or CPU)
                input_tar = tuple([tensor.to(device) for tensor in input_tar])
                input_pre = tuple([tensor.to(device) for tensor in input_pre])
                batch_y = normalizer.norm(batch_y)
                batch_y = batch_y.to(device)
                
                # compute output
                output, syn_descriptor = model(input_tar, input_pre)
                output, log_std = output.chunk(2, dim=1)
                loss = criterion(output, log_std, batch_y)
                val_loss_list.append(loss.data.cpu().numpy()) 
                #pred = normalizer.denorm(output.data.cpu())
        
        val_loss = np.mean(np.array(val_loss_list))
        if (i+1)%10==0:
            print ('Epoch ', i+1, ', training loss: ', train_loss, ', val loss: ',val_loss)
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts) # load best model weights
    print('Training finish')
    
    
    # Test
    model.eval()
    
    pred_value_tr = []
    true_value_tr = []
    syn_descri_tr = []
    idx_tr = []
    pred_value_te = []
    true_value_te = []
    syn_descri_te = []
    idx_te = []
    
    with torch.no_grad(): # Make zero gradient
        for input_tar, input_pre, batch_y, batch_comp, batch_i in train_generator:
            # move tensors to device (GPU or CPU)
            input_tar = tuple([tensor.to(device) for tensor in input_tar])
            input_pre = tuple([tensor.to(device) for tensor in input_pre])
            true = batch_y
            batch_y = normalizer.norm(batch_y)
            batch_y = batch_y.to(device)
            
            # compute output
            output, syn_descriptor = model(input_tar, input_pre)
            output, log_std = output.chunk(2, dim=1)
            pred = normalizer.denorm(output.data.cpu())
            
            pred_value_tr += pred.tolist()
            true_value_tr += true.tolist()
            syn_descri_tr += syn_descriptor.data.cpu().tolist()
            idx_tr += batch_i
           
        for input_tar, input_pre, batch_y, batch_comp, batch_i in test_generator:
            # move tensors to device (GPU or CPU)
            input_tar = tuple([tensor.to(device) for tensor in input_tar])
            input_pre = tuple([tensor.to(device) for tensor in input_pre])
            true = batch_y
            batch_y = normalizer.norm(batch_y)
            batch_y = batch_y.to(device)
            
            # compute output
            output, syn_descriptor = model(input_tar, input_pre)
            output, log_std = output.chunk(2, dim=1)
            pred = normalizer.denorm(output.data.cpu())
            
            pred_value_te += pred.tolist()
            true_value_te += true.tolist()
            syn_descri_te += syn_descriptor.data.cpu().tolist()
            idx_te += batch_i
    
    pred_value_tr = np.array(pred_value_tr)
    true_value_tr = np.array(true_value_tr)
    syn_descri_tr = np.array(syn_descri_tr)
    error_tr = np.abs(pred_value_tr-true_value_tr)
    idx_tr = np.array(idx_tr)
    pred_value_te = np.array(pred_value_te)
    true_value_te = np.array(true_value_te)
    syn_descri_te = np.array(syn_descri_te)
    idx_te = np.array(idx_te)
    print("MAE for test data :", round((sum(np.abs(pred_value_te-true_value_te))/len(pred_value_te))[0], 1))
    
    
    train_val_loss = {'Model_train_loss_curve' : train_loss_curve,
                      'Model_val_loss_curve'   : val_loss_curve,
                       }
    
    pk.dump(idx_te, open('./dataset/test_idx_TPO.sav', 'wb'))
    pk.dump(dataset, open('./dataset/preprocessed_data_TPO.sav', 'wb'))
    
    pk.dump(model, open('./model/trained_model_TPO.sav', 'wb'))
    pk.dump(normalizer, open('./model/fitted_normarlizer_TPO.sav', 'wb'))
    
    pk.dump(train_val_loss, open('./result/train_val_loss_TPO.sav', 'wb'))
    
    
