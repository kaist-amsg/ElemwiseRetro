import json
import copy
import random
import torch
import numpy as np
import pickle as pk
from tqdm import tqdm
from pymatgen.core import Composition
from torch_scatter import scatter_min

from Data import elem_library, get_SourceElem, get_AnionPart
from Train_P import anion_labeling


file_path = "./dataset/InorgSyn_dataset_TP.json"
with open(file_path, "r") as json_file:
    data = json.load(json_file)
file_path = "./dataset/pre_anion_part.json"
with open(file_path, "r") as json_file:
    pre_anion_part = json.load(json_file)
file_path = "./dataset/stoi_dict.json"
with open(file_path, "r") as json_file:
    stoi_dict = json.load(json_file)
file_path = "./dataset/stoi_ll_dict.json"
with open(file_path, "r") as json_file:
    stoi_ll_dict = json.load(json_file)

idx_te = pk.load(open('./dataset/test_idx_TP.sav', 'rb'))

train_data = []
test_data = []
for i in range(len(data)):
    if i in idx_te:
        test_data.append(data[i])
    else:
        train_data.append(data[i])


# Make Statistical counter_part library (popularity-based sampling)
source_elem_dict = {}
for ss in elem_library:
    source_elem_dict[ss] = {}
for i in range(len(train_data)):
    for pre in train_data[i]['Precursors']:
        ca_elem = get_SourceElem([pre])[0]
        if len(ca_elem) != 1:
            print('error')
        ca_elem = ca_elem[0]
        an_elem = get_AnionPart(pre, get_SourceElem([pre])[0])
        
        if an_elem not in source_elem_dict[ca_elem]:
            source_elem_dict[ca_elem][an_elem] = 1
        else:
            source_elem_dict[ca_elem][an_elem] += 1
      
prob_anpart_dict = {}
pre_anion_part_list = list(pre_anion_part)
class_len = len(pre_anion_part_list)
for ss, anpart_count in source_elem_dict.items():
    total_count = 0
    y_label = np.zeros(class_len)
    
    for anpart in anpart_count:
        y_label[pre_anion_part_list.index(anpart)] = anpart_count[anpart]
        total_count += anpart_count[anpart]
    
    if total_count == 0:
        prob_anpart_dict[ss] = np.ones(class_len) / class_len
    else:
        prob_list = y_label/total_count
        prob_anpart_dict[ss] = prob_list

prob_min = 1
for ss in prob_anpart_dict.keys():
    for pp in prob_anpart_dict[ss]:
        if (pp != 0) and (pp < prob_min):
            prob_min = pp

for ss in prob_anpart_dict.keys():
    prob_anpart_dict[ss] += prob_min/10
    prob_anpart_dict[ss] /= sum(prob_anpart_dict[ss])


def data_preprocessing(test_data):
    dataset = []
    for i in range(len(test_data)):
        elements_seq_set = []
        for j in range(len(test_data[i]['Target'])):
            elements_seq = list(Composition(test_data[i]['Target'][j]).get_el_amt_dict().keys())
            elements_seq_set.append(elements_seq)
        source_elem_seq = []
        count=0
        for elem_seq in elements_seq_set:
            for elem in elem_seq:
                if (elem in get_SourceElem(test_data[i]['Target'])[0]) and (elem not in source_elem_seq):
                    source_elem_seq.append(elem)
                    count+=1
        tar_comp = test_data[i]['Target']
        
        y = []
        y_stoi = []
        for elem in source_elem_seq:
            for j in range(len(test_data[i]['Precursors'])):
                if elem in list(Composition(test_data[i]['Precursors'][j]).get_el_amt_dict().keys()):
                    y.append(anion_labeling(test_data[i]['Precursors'][j], pre_anion_part, get_SourceElem(test_data[i]['Target'])[0]))
                    y_stoi.append(test_data[i]['Precursors'][j])
                    
        if len(source_elem_seq) != len(y):
            raise NotImplementedError('labeling error')
        
        y = torch.Tensor(y)
        dataset.append((source_elem_seq, y, y_stoi, tar_comp, i))

    return dataset

def baseline_model(batch):
    output = []
    true = []
    pre_set_idx = []
    for source_elem_seq, y, y_stoi, tar_comp, i in batch:
        for ss in source_elem_seq:
            output.append(random.choices(range(len(list(pre_anion_part))), weights=prob_anpart_dict[ss])[0])
        true.append(y)
        pre_set_idx += [i]*len(source_elem_seq)
    
    output = torch.tensor(np.array(output), dtype=torch.int64)
    true = torch.cat(true, dim=0)
    
    return output, true, pre_set_idx

def baseline_model_stoi(pred, batch):
    batch_source_seq = []
    for i in range(len(batch)):
        elements_seq_set = []
        for j in range(len(batch[i][3])):
            elements_seq = list(Composition(batch[i][3][j]).get_el_amt_dict().keys())
            elements_seq_set.append(elements_seq)
        source_elem_seq = []
        for elem_seq in elements_seq_set:
            for elem in elem_seq:
                if (elem in get_SourceElem(batch[i][3])[0]) and (elem not in source_elem_seq):
                    source_elem_seq.append(elem)
        batch_source_seq += source_elem_seq
    
    pred_precursors = []
    for i in range(len(batch_source_seq)):
        source_part = batch_source_seq[i]
        counter_part = list(pre_anion_part)[pred[i]]
        #stoi_space = stoi_dict[source_part+counter_part]
        stoi_space = stoi_ll_dict[source_part+counter_part]
        if len(stoi_space) == 0:
            pred_precursors.append('['+source_part+']n['+counter_part+']m')
        else:
            #precursor = random.choices(stoi_space)[0]
            precursor = random.choices(list(stoi_space.keys()), weights= list(stoi_space.values()))[0]
            pred_precursors.append(precursor)
    return pred_precursors

def find_top_k_prediction(batch, top_k):
    kth_output = []
    for bb in batch:
        sliced_top_k_result = []
        while len(sliced_top_k_result) < top_k:
            pred, _, _ = baseline_model([bb])
            pred = pred.tolist()
            #if pred not in sliced_top_k_result:
            sliced_top_k_result.append(pred)
        kth_output.append(torch.tensor(sliced_top_k_result))
    return torch.cat(kth_output, dim=1)

dataset = data_preprocessing(test_data)
batch_size = 64
batch_data = [dataset[i *batch_size:(i+1) *batch_size] for i in range((len(dataset)-1 +batch_size)//batch_size)] 


pred_value_te = []
true_value_te = []
pred_stoi_te = []
true_stoi_te = []
pre_set_idx_te = []
top_k_pred_te = []

total_kth_pred_precursors = {}
for k in range(10):
    total_kth_pred_precursors['Top-'+str(k+1)] = []

for batch in tqdm(batch_data):
    pred, true, pre_set_idx = baseline_model(batch)
    true = torch.where(true==1)[1]
    
    true_stoi = []
    for source_elem_seq, y, y_stoi, tar_comp, i in batch:
        true_stoi += y_stoi
    pred_stoi = baseline_model_stoi(pred, batch)
    
    pred_value_te += pred.tolist()
    true_value_te += true.tolist()
    pred_stoi_te += pred_stoi
    true_stoi_te += true_stoi
    
    # compute precursors set index
    pre_set_idx_te.append(torch.tensor(pre_set_idx))
    
    # compute top-k precursors set index
    top_k_pred = find_top_k_prediction(batch, 5)
    top_k_pred_te.append(top_k_pred)
    
    for k in range(top_k_pred.shape[0]):
        if k == 0: #Top-1
            kth_pred_precursors = copy.deepcopy(pred_stoi)
        else:
            kth_pred_precursors = baseline_model_stoi(top_k_pred[k], batch)
        total_kth_pred_precursors['Top-'+str(k+1)] = total_kth_pred_precursors['Top-'+str(k+1)] + kth_pred_precursors


template_pred_value_te = np.array(pred_value_te)
template_true_value_te = np.array(true_value_te)
pre_set_idx_te= torch.cat(pre_set_idx_te, dim=0)
template_top_k_pred_te = torch.cat(top_k_pred_te, dim=1)

accuracy_result = {}

template_te_accuracy = sum(template_pred_value_te == template_true_value_te)/len(template_true_value_te)
print("\nAccuracy for individual_precursor_template of testset :", round(template_te_accuracy,4))
accuracy_result['indiv_template_acc'] = round(template_te_accuracy,4)
for k in range(template_top_k_pred_te.shape[0]):
    if k == 0:
        b = torch.tensor((template_top_k_pred_te[k].numpy() == template_true_value_te), dtype=float)
        te_set_accuracy = sum(scatter_min(torch.tensor((b>0).numpy(), dtype=float), pre_set_idx_te, dim=0)[0])/len(test_data)
        print("\nTop-%d Accuracy for precursors_template_set of testset : %f" %(k+1, round(float(te_set_accuracy),4)))
        accuracy_result['Top-'+str(k+1)+'_template_set_acc'] = round(float(te_set_accuracy),4)
    else:
        b = b + torch.tensor((template_top_k_pred_te[k].numpy() == template_true_value_te), dtype=float)
        te_set_accuracy = sum(scatter_min(torch.tensor((b>0).numpy(), dtype=float), pre_set_idx_te, dim=0)[0])/len(test_data)
        print("Top-%d Accuracy for precursors_template_set of testset : %f" %(k+1, round(float(te_set_accuracy),4)))
        accuracy_result['Top-'+str(k+1)+'_template_set_acc'] = round(float(te_set_accuracy),4)


pk.dump(accuracy_result, open('./result/accuracy_result_TP_baseline.sav', 'wb'))









