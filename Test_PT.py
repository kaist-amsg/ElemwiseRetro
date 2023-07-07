import json
import random
import math
import sys
import numpy as np
import pandas as pd
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_mul
from torch.utils.data import DataLoader

from tqdm import tqdm
from pymatgen.core import Composition

from Data import get_SourceElem, get_AnionPart, get_Source_Anion_ratio, get_tar_elementdist
from Model import collate_batch, collate_batch2
from Train_P import find_precursors_set, find_precursors_indiv
from Train_T import find_syn_temperature

random.seed(8888)
torch.manual_seed(8888)
np.random.seed(8888)


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
training_type = input("Which model to test ? (RandSplit or TimeSplit) = ")
if training_type not in ['RandSplit', 'TimeSplit']:
    print("Invalid name of model type !")
    sys.exit()
if training_type == 'RandSplit':
    check_time_transferability = False
else:
    check_time_transferability = True


if check_time_transferability == False:
    print("-----------Testing RandSplit case-----------")
    file_path = "./dataset/InorgSyn_dataset_TP.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
else:
    print("-----------Testing TimeSplit case-----------")
    file_path = "./dataset/InorgSyn_dataset_TP2.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    from datetime import datetime
    pubyear_count_dict = {}
    for dd in data:
        if dd['pubdate'] != 'N/A':
            pub_date1 = datetime.strptime(dd['pubdate'], "%Y-%m-%d")
            pubyear = pub_date1.year
            if pubyear in pubyear_count_dict.keys():
                pubyear_count_dict[pubyear] += 1
            else:
                pubyear_count_dict[pubyear] = 1

    pubyear_list = list(pubyear_count_dict.keys())
    pubyear_list.sort()
    count = []
    for yr in pubyear_list:
        count.append(pubyear_count_dict[yr])
    
    year_cutoff = 2016
    tr_count = 0
    te_count = 0
    for yr in pubyear_list:
        if yr < year_cutoff:
            tr_count += pubyear_count_dict[yr]
        else:
            te_count += pubyear_count_dict[yr]
    print("(TimeSplit) Train/Test = ", tr_count/(tr_count+te_count), te_count/(tr_count+te_count))

    pubyear_list_2 = [yr for yr in pubyear_list if yr%2==0]

    plt.figure(figsize=(22,8))
    plt.bar(pubyear_list, count, width=0.6 ,color=['gray']*pubyear_list.index(year_cutoff)+['tab:blue']*(len(pubyear_list)-pubyear_list.index(year_cutoff)))
    plt.gca().spines['left'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().tick_params(width=2.5, length=8)
    plt.xlabel('Year', size=35, labelpad=15)
    plt.ylabel('Counts', size=40, labelpad=15)
    plt.xticks(pubyear_list_2, rotation=0, size=30)
    plt.yticks(size=30)
    plt.xlim([1999,2021])
    plt.ylim([0,1300])
    plt.show()

file_path = "./dataset/pre_anion_part.json"
with open(file_path, "r") as json_file:
    pre_anion_part = json.load(json_file)
file_path = "./dataset/stoi_dict.json"
with open(file_path, "r") as json_file:
    stoi_dict = json.load(json_file)

get_tar_elementdist(data)  # Figure 1. Map of the elements for the target inorganic compositions


if check_time_transferability == False:
    idx_te = pk.load(open('./dataset/test_idx_TP.sav', 'rb'))
    dataset = pk.load(open('./dataset/preprocessed_data_TP.sav', 'rb'))
    accuracy_result = pk.load(open('./result/accuracy_result_TP_FalseFalseFalse.sav', 'rb'))
else:
    idx_te = pk.load(open('./dataset/test_idx_TP_time.sav', 'rb'))
    dataset = pk.load(open('./dataset/preprocessed_data_TP_time.sav', 'rb'))
    accuracy_result = pk.load(open('./result/accuracy_result_TP_FalseFalseFalse_time.sav', 'rb'))
idx_te_T = pk.load(open('./dataset/test_idx_TPO.sav', 'rb'))
dataset_T = pk.load(open('./dataset/preprocessed_data_TPO.sav', 'rb'))
accuracy_result_1 = pk.load(open('./result/accuracy_result_TP_FalseFalseTrue.sav', 'rb'))
accuracy_result_2 = pk.load(open('./result/accuracy_result_TP_FalseTrueTrue.sav', 'rb'))
accuracy_result_3 = pk.load(open('./result/accuracy_result_TP_TrueFalseTrue.sav', 'rb'))
accuracy_result_4 = pk.load(open('./result/accuracy_result_TP_baseline.sav', 'rb'))

print(accuracy_result)     # Table 1. ElemwiseRetro
print(accuracy_result_1)   # Table 2. Source elem-wise
print(accuracy_result_2)   # Table 2. Source elem-wise w. GLA
print(accuracy_result_3)   # Table 2. Global agg.
print(accuracy_result_4)   # baseline Model (Statistical sampling)


if check_time_transferability == False:
    TP_train_val_loss = pk.load(open('./result/train_val_loss_TP_FalseFalseFalse.sav', 'rb'))
else:
    TP_train_val_loss = pk.load(open('./result/train_val_loss_TP_FalseFalseFalse_time.sav', 'rb'))
TP_model_train_loss_curve = TP_train_val_loss['Model_train_loss_curve']
TP_model_val_loss_curve  = TP_train_val_loss['Model_val_loss_curve']
TPO_train_val_loss = pk.load(open('./result/train_val_loss_TPO.sav', 'rb'))
TPO_model_train_loss_curve = TPO_train_val_loss['Model_train_loss_curve']
TPO_model_val_loss_curve  = TPO_train_val_loss['Model_val_loss_curve']

def draw_trainingloss_curve():
    plt.figure(figsize=(10,10))
    epoch_list = np.arange(1,len(TP_model_train_loss_curve[:30])+1)
    plt.plot(epoch_list, TP_model_train_loss_curve[:30], label='Train loss', linewidth=5)
    plt.plot((epoch_list+1)[:29], TP_model_val_loss_curve[:29], label='Val loss', linewidth=5)
    plt.gca().spines['left'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().tick_params(width=2.5, length=8)
    plt.title("Training loss curve", fontsize=40, pad=30)
    plt.xlabel('Epoch', fontsize=40, labelpad=15)
    plt.ylabel('Loss', fontsize=40, labelpad=15)
    plt.legend(fontsize=30)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.show()

    plt.figure(figsize=(10,10))
    epoch_list = np.arange(1,len(TPO_model_train_loss_curve[:50])+1)
    plt.plot(epoch_list, TPO_model_train_loss_curve[:50], label='Train loss', linewidth=5)
    plt.plot((epoch_list+1)[:49], TPO_model_val_loss_curve[:49], label='Val loss', linewidth=5)
    plt.gca().spines['left'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().tick_params(width=2.5, length=8)
    plt.title("Training loss curve", fontsize=40, pad=30)
    plt.xlabel('Epoch', fontsize=40, labelpad=15)
    plt.ylabel('Loss', fontsize=40, labelpad=15)
    plt.legend(fontsize=30)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.ylim([0.1, 1.05])
    plt.show()

draw_trainingloss_curve()     # Figure S(a). ElemwiseRetro & TempPrediction training loss curve


# Prepare model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if check_time_transferability == False:
    model = pk.load(open('./model/trained_model_TP_FalseFalseFalse.sav', 'rb'))
else:
    model = pk.load(open('./model/trained_model_TP_FalseFalseFalse_time.sav', 'rb'))
model.to(device)
normalizer = pk.load(open('./model/fitted_normarlizer_TPO.sav', 'rb'))
model_T = pk.load(open('./model/trained_model_TPO.sav', 'rb'))
model_T.to(device)


def draw_elemnum_accuracy_plot():
    if check_time_transferability == False:
        file_path = "./dataset/InorgSyn_dataset_TP.json"
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        idx_te = pk.load(open('./dataset/test_idx_TP.sav', 'rb'))
        dataset = pk.load(open('./dataset/preprocessed_data_TP.sav', 'rb'))
    else:
        file_path = "./dataset/InorgSyn_dataset_TP2.json"
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        idx_te = pk.load(open('./dataset/test_idx_TP_time.sav', 'rb'))
        dataset = pk.load(open('./dataset/preprocessed_data_TP_time.sav', 'rb'))

    test_set = []
    for idx in idx_te:
        test_set.append(dataset[idx])

    data_params = {"batch_size": 128, "num_workers": 0, "pin_memory": False,
                   "shuffle": False, "collate_fn": collate_batch}

    test_idx = list(range(len(test_set)))
    test_set = torch.utils.data.Subset(test_set, test_idx[0::1])
    test_generator = DataLoader(test_set, **data_params)

    sourceElem_num_accuracy_dict = {}
    sourceElem_num_count_dict = {}

    model.eval()
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

            #score = torch.kthvalue(F.softmax(template_output, dim=1), template_output.shape[1])[0].cpu()
            pred = template_output.max(dim=1)[1].cpu()
            true = batch_y.cpu()
            pre_set_idx = pre_set_idx.cpu()

            set_pred_true = scatter_mean(torch.tensor((pred==true), dtype=float), pre_set_idx, dim=0)

            for ii, idx in enumerate(batch_i):
                sourceElem_num = len(data[idx]['Precursors'])
                if sourceElem_num == 1:
                    pass
                else:
                    if sourceElem_num >= 6:
                        sourceElem_num = "Above 6"
                    else:
                        sourceElem_num = str(sourceElem_num)
    
                    if sourceElem_num not in sourceElem_num_accuracy_dict:
                        sourceElem_num_accuracy_dict[sourceElem_num] = 0
                    else:
                        if set_pred_true[ii] == 1:
                            sourceElem_num_accuracy_dict[sourceElem_num] += 1
    
                    if sourceElem_num not in sourceElem_num_count_dict:
                        sourceElem_num_count_dict[sourceElem_num] = 0
                    else:
                        sourceElem_num_count_dict[sourceElem_num] += 1

    sourceElem_num_accuracy_dict = dict(sorted(sourceElem_num_accuracy_dict.items()))
    sourceElem_num_count_dict = dict(sorted(sourceElem_num_count_dict.items()))

    elem_num_list = []
    accuracy_list = []
    count_list = []
    for key, value in sourceElem_num_accuracy_dict.items():
        sourceElem_num_accuracy_dict[key] = value/sourceElem_num_count_dict[key]

        elem_num_list.append(str(key))
        accuracy_list.append(sourceElem_num_accuracy_dict[key])
        count_list.append(sourceElem_num_count_dict[key])


    data = {'Elem number': elem_num_list, 'Accuracy': accuracy_list, 'Counts': count_list}
    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.set_title('Set of precursors prediction', fontsize=40, pad=30)
    color = 'tab:red'
    ax1.set_xlabel('Number of source elements', fontsize=40, labelpad=15)
    ax1.set_ylabel('Accuracy', fontsize=45, color=color, labelpad=15)
    ax1 = sns.lineplot(x='Elem number', y='Accuracy', data = df, sort=False, color=color, marker='o', linewidth=3, markersize=12)
    ax1.tick_params(axis='x', width=2.5, length=8, labelsize=30)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_ylim([0,1])
    ax1.tick_params(axis='y', color=color, width=2.5, length=8, labelsize=30)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['right'].set_linewidth(2.5)
    ax1.spines['top'].set_linewidth(2.5)
    ax1.spines['bottom'].set_linewidth(2.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Counts', fontsize=45, color=color, labelpad=15)
    ax2 = sns.barplot(x='Elem number', y='Counts', data = df, color='tab:blue')
    ax2.tick_params(axis='y', width=2.5, length=8, labelsize=30)
    ax2.set_ylim([0, 800])
    if check_time_transferability == False:
        ax2.set_ylim([0, 800])
    else:
        ax2.set_ylim([0, 1600])
    plt.show()

draw_elemnum_accuracy_plot()


def draw_score_accuracy_plot():
    if check_time_transferability == False:
        idx_te = pk.load(open('./dataset/test_idx_TP.sav', 'rb'))
        dataset = pk.load(open('./dataset/preprocessed_data_TP.sav', 'rb'))
    else:
        idx_te = pk.load(open('./dataset/test_idx_TP_time.sav', 'rb'))
        dataset = pk.load(open('./dataset/preprocessed_data_TP_time.sav', 'rb'))

    test_set = []
    for idx in idx_te:
        test_set.append(dataset[idx])

    data_params = {"batch_size": 128, "num_workers": 0, "pin_memory": False,
                   "shuffle": False, "collate_fn": collate_batch}

    test_idx = list(range(len(test_set)))
    test_set = torch.utils.data.Subset(test_set, test_idx[0::1])
    test_generator = DataLoader(test_set, **data_params)

    grid_num = 10
    set_grid_num = 10
    s_score = []
    for i in range(grid_num):
        label = str(i/grid_num)+'~'+str((i+1)/grid_num)
        s_score.append(label)
    score_count = [0]*grid_num
    score_accuracy = [0]*grid_num

    set_s_score = []
    for i in range(set_grid_num):
        label = str(i/set_grid_num)+'~'+str((i+1)/set_grid_num)
        set_s_score.append(label)
    set_score_count = [0]*set_grid_num
    set_score_accuracy = [0]*set_grid_num

    model.eval()
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

            score = torch.kthvalue(F.softmax(template_output, dim=1), template_output.shape[1])[0].cpu()
            pred = template_output.max(dim=1)[1].cpu()
            true = batch_y.cpu()
            pre_set_idx = pre_set_idx.cpu()

            for i in range(len(score)):
                if score[i] == 1:
                    score_idx = grid_num-1
                else:
                    score_idx = math.trunc(score[i].item()*grid_num)
                score_count[score_idx] += 1
                if pred[i] == true[i]:
                    score_accuracy[score_idx] += 1

            set_pred_true = scatter_mean(torch.tensor((pred==true), dtype=float), pre_set_idx, dim=0)
            set_score = scatter_mul(score, pre_set_idx, dim=0)
            for i in range(len(set_score)):
                if set_score[i] == 1:
                    score_idx = set_grid_num-1
                else:
                    score_idx = math.trunc(set_score[i].item()*set_grid_num)
                set_score_count[score_idx] += 1
                if set_pred_true[i] == 1:
                    set_score_accuracy[score_idx] += 1

    s_score = s_score[math.ceil(grid_num*4/10)+1:]
    s_count = score_count[math.ceil(grid_num*4/10)+1:]
    s_accuracy = score_accuracy[math.ceil(grid_num*4/10)+1:]
    s_accuracy = np.array(s_accuracy)/np.array(s_count)

    set_s_score = set_s_score[math.ceil(set_grid_num*3/10):]
    set_s_count = set_score_count[math.ceil(set_grid_num*3/10):]
    set_s_accuracy = set_score_accuracy[math.ceil(set_grid_num*3/10):]
    set_s_accuracy = np.array(set_s_accuracy)/np.array(set_s_count)

    data = {'Score': s_score, 'Accuracy': s_accuracy, 'Counts': s_count}
    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.set_title('Individual precursors prediction', fontsize=40, pad=30)
    color = 'tab:red'
    ax1.set_xlabel('Prediction score', fontsize=40, labelpad=15)
    ax1.set_ylabel('Accuracy', fontsize=45, color=color, labelpad=15)
    ax1 = sns.lineplot(x='Score', y='Accuracy', data = df, sort=False, color=color, marker='o', linewidth=3, markersize=12)
    ax1.tick_params(axis='x', rotation=55, width=2.5, length=8, labelsize=30)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_ylim([-0.05,1])
    ax1.tick_params(axis='y', color=color, width=2.5, length=8, labelsize=30)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['right'].set_linewidth(2.5)
    ax1.spines['top'].set_linewidth(2.5)
    ax1.spines['bottom'].set_linewidth(2.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Counts', fontsize=45, color=color, labelpad=15)
    ax2 = sns.barplot(x='Score', y='Counts', data = df, color='tab:blue')
    ax2.tick_params(axis='y', width=2.5, length=8, labelsize=30)
    if check_time_transferability == False:
        ax2.set_ylim([0, 4000])
    else:
        ax2.set_ylim([0, 10000])
    plt.show()

    data = {'Score': set_s_score, 'Accuracy': set_s_accuracy, 'Counts': set_s_count}
    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.set_title('Set of precursors prediction', fontsize=40, pad=30)
    color = 'tab:red'
    ax1.set_xlabel('Prediction score', fontsize=40, labelpad=15)
    ax1.set_ylabel('Accuracy', fontsize=45, color=color, labelpad=15)
    ax1 = sns.lineplot(x='Score', y='Accuracy', data = df, sort=False, color=color, marker='o', linewidth=3, markersize=12)
    ax1.tick_params(axis='x', rotation=55, width=2.5, length=8, labelsize=30)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_ylim([-0.05,1])
    ax1.tick_params(axis='y', color=color, width=2.5, length=8, labelsize=30)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['right'].set_linewidth(2.5)
    ax1.spines['top'].set_linewidth(2.5)
    ax1.spines['bottom'].set_linewidth(2.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Counts', fontsize=45, color=color, labelpad=15)
    ax2 = sns.barplot(x='Score', y='Counts', data = df, color='tab:blue')
    ax2.tick_params(axis='y', width=2.5, length=8, labelsize=30)
    if check_time_transferability == False:
        ax2.set_ylim([0, 600])
    else:
        ax2.set_ylim([0, 1500])
    plt.show()

draw_score_accuracy_plot()     # Figure 3. Prediction score vs prediction accuracy plot


def draw_temp_parityplot(model_T,normalizer,device):
    dataset_te_T = []
    for idx in idx_te_T:
        dataset_te_T.append(dataset_T[idx])

    input_tar, input_pre, batch_y, batch_comp, batch_i = collate_batch2(dataset_te_T)

    model_T.eval()

    pred_value_te = []
    true_value_te = []
    syn_descri_te = []
    with torch.no_grad(): # Make zero gradient
        input_tar = tuple([tensor.to(device) for tensor in input_tar])
        input_pre = tuple([tensor.to(device) for tensor in input_pre])
        true = batch_y
        batch_y = normalizer.norm(batch_y)
        batch_y = batch_y.to(device)

        # compute output
        output, syn_descriptor = model_T(input_tar, input_pre)
        output, log_std = output.chunk(2, dim=1)
        pred = normalizer.denorm(output.data.cpu())

        pred_value_te += pred.tolist()
        true_value_te += true.tolist()
        syn_descri_te += syn_descriptor.data.cpu().tolist()

    pred_value_te = np.array(pred_value_te)
    true_value_te = np.array(true_value_te)
    syn_descri_te = np.array(syn_descri_te)
    print("MAE for test data :", round((sum(np.abs(pred_value_te-true_value_te))/len(pred_value_te))[0], 1))

    #fig = plt.figure(figsize=(6,6))
    import matplotlib as mpl
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(10,10))
    x = true_value_te.T[0]
    y = pred_value_te.T[0]
    xedges = np.arange(temp_min, temp_max+1, (temp_max-temp_min)/25)
    yedges = np.arange(temp_min, temp_max+1, (temp_max-temp_min)/25)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    cmap = mpl.colormaps['Blues']
    plt.imshow(H.T, origin='lower',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], cmap=cmap, norm=LogNorm())
    #plt.title("2d histogram for test", fontsize=25)
    plt.gca().spines['left'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().tick_params(width=2.5, length=8)
    plt.xlabel('Labeled temperature (℃)', size=35, labelpad=15)
    plt.ylabel('Predicted temperature (℃)', size=35, labelpad=15)
    plt.xticks(fontsize=30, rotation=45)
    plt.yticks(fontsize=30, rotation=45)
    cbar = plt.colorbar(pad=0.1, fraction=0.043, ticks=[10**0,10**0.5,10**1])
    cbar.ax.set_yticklabels(['$10^{0}$','$10^{0.5}$','$10^{1}$'])
    cbar.ax.tick_params(labelsize=27, width=2.5, length=8)
    cbar.outline.set_linewidth(2.5)
    plt.show()

draw_temp_parityplot(model_T,normalizer,device)  # Figure 4.


# Show model result
def Show_LLZO_prediction():
    target_tar = ["Li7La3Zr2O12"]
    print("Predicted precursors :")
    for pre in find_precursors_indiv(target_tar, 5, model,device,embedding_dict,pre_anion_part,stoi_dict):
        print(pre)
    print('\n')
    for pre in find_precursors_set(target_tar, 5, model,device,embedding_dict,pre_anion_part,stoi_dict):
        print(pre)
    print('\n')

    print("Labeled precursors :")
    for i in range(len(data)):
        tar_list = data[i]['Target']
        tar_list.sort()
        tar = str(tar_list)

        target_tar.sort()
        target_t = str(target_tar)

        if tar == target_t:
            print(data[i]['Precursors'])

Show_LLZO_prediction()     # Figure S1. LLZO prediction by ElemwiseRetro


