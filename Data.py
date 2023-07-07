import json
import re
import math
import numpy as np
import operator
import requests
import time
from tqdm import tqdm
from pymatgen.core import Composition

import torch
import matplotlib.pyplot as plt


def doi2pubdate(data, your_scopus_apikey):
    def preparing_text_dataset(url, h):
        try:
            page_request = requests.get(url, headers=h)
            count = 0
            while page_request.status_code != 200:
                page_request = requests.get(url, headers=h)
                count += 1
                if count == 100:
                    print("Infinite recursive..")
                time.sleep(0.2)
            if count >= 100:
                print("Infinite recursive escape!")
            page = json.loads(page_request.content.decode("utf-8-sig"))
            articles_list = page['search-results']['entry']
            titles = []
            pubdates = []
            dois = []
            abstracts = []
            for article in articles_list:
                if article.get('dc:title'):
                    title = article['dc:title']
                else:
                    title = "N/A"
                if article.get('prism:coverDate'):
                    pubdate = article['prism:coverDate']
                else:
                    pubdate = "N/A"
                if article.get('prism:doi'):
                    doi = article['prism:doi']
                else:
                    doi = "N/A"
                if article.get('dc:description'):
                    abstract = article['dc:description']
                else:
                    abstract = "N/A"
                #
                titles.append(title)
                pubdates.append(pubdate)
                dois.append(doi)
                abstracts.append(abstract)
        except:
            print("\nRequests error, ", url, "\n")
            titles = ["N/A"]
            pubdates = ["N/A"]
            dois = ["N/A"]
            abstracts = ["N/A"]
        try:
            item_num = int(page['search-results']['opensearch:itemsPerPage'])
        except:
            print("\nItem_num error\n")
            item_num = 1 #마지막 줄에 "N/A"로 한 줄 추가하기 위해
        try:
            total_num = int(page['search-results']['opensearch:totalResults'])
        except:
            print("\nTotal_num error\n")
            total_num = 9999999
        try:
            link_list = page['search-results']['link']
            next_url = "N/A"
            for link in link_list:
                if link['@ref'] == 'next':
                    next_url = link['@href']
        except:
            print("\nNext_url error\n")
            next_url = "N/A"
        try:
            remaining_num = int(page_request.headers["X-RateLimit-Remaining"])
            reset_time = time.ctime(int(str(page_request.headers["X-RateLimit-Reset"])[:-3]))
        except:
            print("\nRemaining_num & reset_time error\n")
            remaining_num = "Error"
            reset_time = "Error"
            #
        return titles, pubdates, dois, abstracts, item_num, total_num, next_url, remaining_num, reset_time

    def checking_apikey_remaining(url, h):
        page_request = requests.get(url, headers=h)
        try:
            remaining_num = int(page_request.headers["X-RateLimit-Remaining"])
        except:
            remaining_num = 0
        if remaining_num < 10:
            check = True
        else:
            check = False
        return check

    def encoding_keyword(keyword):
        encoded_keyword = re.sub("[(]", "%28", keyword)
        encoded_keyword = re.sub("[)]", "%29", encoded_keyword)
        encoded_keyword = re.sub("[/]", "%2f", encoded_keyword)
        return encoded_keyword

    api_resource = "http://api.elsevier.com/content/search/scopus"

    # headers
    headers = {}
    apikey_list = [your_scopus_apikey]
    ith_apikey = 0
    headers['X-ELS-APIKey'] = apikey_list[ith_apikey]
    headers['X-ELS-ResourceVersion'] = 'XOCS'
    headers['Accept'] = 'application/json'

    remaining_num = 0

    result = []
    for dd in tqdm(data):
        keyword = "DOI" + "(" + dd['doi'] + ")"
        encoded_keyword = encoding_keyword(keyword+ '+AND+(DOCTYPE(ar))')   # Article document type only
        query = "?query=" + encoded_keyword   #"KEYWORD" in author keyword
        cursor = "&cursor=*&count=25"
        condition = "&view=complete&sort=pubyear"
        field = "&field=doi,title,description,coverDate"
        #date = "&date=1900-2019"
        url = api_resource + query + cursor + condition + field
        
        if remaining_num < 10:
            while checking_apikey_remaining(url, headers):
                ith_apikey += 1
                time.sleep(0.1)
                if len(apikey_list) > ith_apikey:
                    headers['X-ELS-APIKey'] = apikey_list[ith_apikey]
                    print("Changing API key")
                else:
                    print("End API key list")
                    break

        titles, pubdates, dois, abstracts, item_num, total_num, url, remaining_num, reset_time = preparing_text_dataset(url, headers)

        rr = {}
        rr['Target'] = dd['Target']
        rr['Precursors'] = dd['Precursors']
        rr['doi'] = dd['doi']
        if (len(dois) == 1) and (dois[0] == dd['doi']) and (len(pubdates) == 1):
            rr['pubdate'] = pubdates[0]
        else:
            rr['pubdate'] = 'N/A'
        result.append(rr)

    return result

def get_ordered_syn_elem_library(data):

    def get_ElemCountDict(data):
        elem_count_dict = {}
        for i in range(len(data['reactions'])):
            for j in range(len(data['reactions'][i]['target']['composition'])):
                for key, value in data['reactions'][i]['target']['composition'][j]['elements'].items():
                    if key not in elem_count_dict.keys():
                        elem_count_dict[key] = 1
                    elif key in elem_count_dict.keys():
                        elem_count_dict[key] += 1
                    else:
                        print('error')
        #print(elem_count_dict)
        return elem_count_dict

    elem_count_dict = get_ElemCountDict(data)
    total_elem = [alkali_metal, alkaline_earth_metal, transition_metal, lanthanide_elem, actinide_elem, post_transition_metal, metalloid, non_metal, noble_gas, artificial_elem]

    #print(len(elem_library)) # 118

    syn_elem_library = []
    unsyn_elem_library = []
    for key, value in elem_count_dict.items():
        if (key in elem_library) and (value > 0):
            syn_elem_library.append(key)
        else:
            unsyn_elem_library.append(key)

    ordered_syn_elem_library = []
    for elem_group in total_elem:
        for i in elem_group:
            if i in syn_elem_library:
                ordered_syn_elem_library.append(i)

    return ordered_syn_elem_library

def get_tar_elementdist(data_TP):

    def key_count_append(elem_c_dict, elem_list):
        for elem in elem_list:
            if elem in elem_c_dict:
                elem_c_dict[elem] += 1
            else:
                elem_c_dict[elem] = 1
        return elem_c_dict

    source_elem_count_dict = {}
    anion_elem_count_dict = {}
    for i in range(len(data_TP)):
        elements_seq_set = []
        for j in range(len(data_TP[i]['Target'])):
            elem_list = list(Composition(data_TP[i]['Target'][j]).get_el_amt_dict().keys())
            elements_seq_set += elem_list
        elements_seq_set = list(set(elements_seq_set))

        t_source_elem, t_env_elem = get_SourceElem(data_TP[i]['Target'], comp_type='Target')

        source_elem_count_dict = key_count_append(source_elem_count_dict, t_source_elem)
        anion_elem_count_dict = key_count_append(anion_elem_count_dict, t_env_elem)

    log_source_elem_count_dict = {}
    log_anion_elem_count_dict = {}
    for elem in source_elem_count_dict.keys():
        log_source_elem_count_dict[elem] = math.log(source_elem_count_dict[elem], 10)
    for elem in anion_elem_count_dict.keys():
        log_anion_elem_count_dict[elem] = math.log(anion_elem_count_dict[elem], 10)

    import Util
    Util.PlotPTable2(log_source_elem_count_dict)
    Util.PlotPTable2(log_anion_elem_count_dict)
    log_source_elem_count_dict.update(log_anion_elem_count_dict)
    Util.PlotPTable2(log_source_elem_count_dict)

def get_SourceElem(comp_list, comp_type='Target'):
    source_elem = []
    env_elem = []
    for comp in comp_list:
        non_source_elem = []
        comp_dict = Composition(comp).get_el_amt_dict()
        elements_seq = list(comp_dict.keys())

        for ee in elements_seq:
            if ee in essen_elem:
                source_elem.append(ee)
            else:
                non_source_elem.append(ee)
        for ee in non_source_elem:
            env_elem.append(ee)

    source_elem = list(set(source_elem))
    env_elem = list(set(env_elem))

    return source_elem, env_elem

def get_Source_Anion_ratio(comp_list):
    ratio_list = []
    for vv in list(comp_list):
        p_source_elem, _ = get_SourceElem([vv])
        source_part = p_source_elem[0]
        template = get_AnionPart(vv, p_source_elem)
        ca_ratio = Composition(vv).get_el_amt_dict()[source_part]
        s_removed_comp = ''
        comp_dict = Composition(vv).get_el_amt_dict()
        for elem, stoi in comp_dict.items():
            if elem != source_part:
                s_removed_comp += elem+str(stoi)
        if s_removed_comp == '':
            an_ratio = 0
        else:
            if Composition(s_removed_comp).get_integer_formula_and_factor()[0] == Composition(template).get_integer_formula_and_factor()[0]:
                an_ratio = Composition(s_removed_comp).get_integer_formula_and_factor()[1] / Composition(template).get_integer_formula_and_factor()[1]
            else:
                raise NotImplementedError()
        ca_an_ratio = an_ratio/ca_ratio
        if ca_an_ratio not in ratio_list:
            ratio_list.append(ca_an_ratio)
    return ratio_list

def select_CorrectlyParsedSynData(data, ordered_syn_elem_library):

    def rangedtemp_to_avgtemp(a):
        temp_list = []
        temp = None
        unit = 'Unlabeled'

        if a != None:
            for n in range(len(a)):
                if len(a[n]['values']) > 0:
                    temp_list += a[n]['values']
                    unit = a[n]['units']
                elif (len(a[n]['values']) == 0):
                    if (a[n]['max_value'] != None):
                        temp_list += [a[n]['max_value']]
                        unit = a[n]['units']
                    if (a[n]['min_value'] != None):
                        temp_list += [a[n]['min_value']]
                        unit = a[n]['units']
            if (len(temp_list)>0):
                temp = np.mean(np.array(temp_list)) # averaging
            if (unit != 'Unlabeled') and ('C' not in unit):
                temp = temp - 273
                unit = 'C'

        return temp, unit

    result = []
    result_for_PreTar = []
    filtered_data = []

    for i in range(len(data['reactions'])):
        syn = {}
        syn['Target'] = []
        syn['Precursors'] = []
        syn['Operation'] = []
        syn['doi'] = data['reactions'][i]['doi']

        syn_TP = {}
        syn_TP['Target'] = []
        syn_TP['Precursors'] = []
        syn_TP['doi'] = data['reactions'][i]['doi']

        # Target parsing
        if len(data['reactions'][i]['target']['composition']) >= 1:
            for ii in range(len(data['reactions'][i]['target']['composition'])):
                tar_composition = data['reactions'][i]['target']['composition'][ii]['formula']
                try:
                    comp = Composition(str(tar_composition))
                    if len(comp.get_el_amt_dict()) != 0:
                        check = True
                        for elem in comp.get_el_amt_dict().keys():
                            if elem not in ordered_syn_elem_library:
                                check = False
                        if check:
                            syn['Target'].append(str(tar_composition))
                except:
                    # x, y, z in stoi case
                    element = data['reactions'][i]['target']['composition'][ii]['elements']
                    amount_var = data['reactions'][i]['target']['amounts_vars']
                    try:
                        tar_compound_name = ""
                        for elem, stoi in element.items():
                            if re.search("[a-zA-Z]", stoi) != None:
                                check = True
                                var_s = re.findall("[a-zA-Z]", stoi)
                                for var in var_s:
                                    if len(amount_var[var]['values']) != 0:
                                        stoi = re.sub(var, str(round(np.mean(amount_var[var]['values']),3)), stoi)
                                    elif (amount_var[var]['max_value'] != None) and (amount_var[var]['min_value'] != None):
                                        stoi = re.sub(var, round((amount_var[var]['max_value']+amount_var[var]['min_value'])/2,3), stoi)
                                    elif amount_var[var]['max_value'] != None:
                                        stoi = re.sub(var, round(amount_var[var]['max_value'],3), stoi)
                                    elif amount_var[var]['min_value'] != None:
                                        stoi = re.sub(var, round(amount_var[var]['min_value'],3), stoi)
                                    else:
                                        check = False
                                if check:
                                    stoi = eval(stoi)
                                    if (round(stoi,3) > 15) or (round(stoi,3) < 0):
                                        raise NotImplementedError()
                                    stoi = str(round(stoi,3))
                            if stoi != '0':
                                tar_compound_name += elem + stoi

                        comp = Composition(str(tar_compound_name))
                        if len(comp.get_el_amt_dict()) != 0:
                            check = True
                            for elem in comp.get_el_amt_dict().keys():
                                if elem not in ordered_syn_elem_library:
                                    check = False
                            if check:
                                comp = Composition(str(tar_compound_name))
                                syn['Target'].append(str(tar_compound_name))
                    except:
                        pass # skip vague composition cases

        # Precursors parsing
        for j in range(len(data['reactions'][i]['precursors'])):
            if len(data['reactions'][i]['precursors'][j]['composition']) == 1:
                pre_composition = data['reactions'][i]['precursors'][j]['composition'][0]['formula']
                try:
                    pre_composition = re.sub('[^()1-9]?[1-9]?H2O', '', pre_composition)
                    comp = Composition(str(pre_composition))
                    if len(comp.get_el_amt_dict()) != 0:
                        check = True
                        for elem, stoi in comp.items():
                            if str(elem) not in ordered_syn_elem_library:
                                check = False
                        if str(pre_composition) == 'FeC2O4.2H20': pre_composition = 'FeC2O4'
                        if str(pre_composition) in ['C4H6Mn','PO(OC4H9)4','Fe(CH3CHOHCOO)2','CoCO3.3Co(OH)2']: check = False
                        if check:
                            syn['Precursors'].append(str(pre_composition))
                except:
                    pass # skip vague composition cases

            elif len(data['reactions'][i]['precursors'][j]['composition']) > 1:
                pass # skip vague or multi_counter_part cases

        # Operation parsing
        for j in range(len(data['reactions'][i]['operations'])):
            a = data['reactions'][i]['operations'][j]['type']
            T = 0
            if a == 'StartingSynthesis':    a = 'Start'
            elif a == 'HeatingOperation':   a = 'Heat'
            elif a == 'QuenchingOperation': a = 'Quench'
            elif a == 'DryingOperation':    a = 'Dry'
            elif a == 'MixingOperation':
                a = 'Mix'
            elif a == 'ShapingOperation':
                a = 'Shape'
            T = data['reactions'][i]['operations'][j]['conditions']['heating_temperature']
            T, u = rangedtemp_to_avgtemp(T) # unit : Celsius
            if T != None:
                if (T<=300)or(T>1600)or(a=='Start')or(a=='Quench')or(a=='Dry')or(a=='Mix')or(a=='Shape'):
                    T = None
            if T != None:
                syn['Operation'].append([a, round(T,1)])

        # Collect Data
        if (len(syn['Target'])!=0) and (len(syn['Precursors']) not in [0, 1]) and (len(syn['Operation']) != 0):
            result.append(syn)
        else:
            filtered_data.append(syn)

        # dataset which only contains Target & Precursors
        if (len(syn['Target'])!=0) and (len(syn['Precursors']) not in [0, 1]):
            syn_TP['Target'] = syn['Target']
            syn_TP['Precursors'] = syn['Precursors']
            result_for_PreTar.append(syn_TP)

    return result, filtered_data, result_for_PreTar

def select_MassConservation(PreparedData):
    """
    1. Remove additive (e.g., NH4OH) or gas (e.g., O2, H2O)
    2. Target_Source_Element_set == Precursors_Source_Element_set
    """
    result = []
    filtered_data = []
    for i in range(len(PreparedData)):
        for t in PreparedData[i]['Target']:
            t_source_elem, _ = get_SourceElem([t], comp_type='Precursor')
            if len(t_source_elem) == 0:
                PreparedData[i]['Target'].remove(t)
        for p in PreparedData[i]['Precursors']:
            p_source_elem, _ = get_SourceElem([p], comp_type='Precursor')
            if len(p_source_elem) == 0:
                PreparedData[i]['Precursors'].remove(p)

        tar_source_elem, _ = get_SourceElem(PreparedData[i]['Target'], comp_type='Target')
        pre_source_elem, _ = get_SourceElem(PreparedData[i]['Precursors'], comp_type='Target')
        """
        for j in range(len(PreparedData[i]['Precursors'])):
            comp = Composition(PreparedData[i]['Precursors'][j])
            for elem, stoi in comp.items():
                if str(elem) in source_elem:
                    pre_source_elem.append(str(elem))
        """

        if (set(tar_source_elem)==set(pre_source_elem)):
            result.append(PreparedData[i])
        else:
            filtered_data.append(PreparedData[i])

    return result, filtered_data

def select_CommerciallyViable(PreparedData):
    """
    Identify precursors which have their CAS number.
    """
    #file_path = "./dataset/CAS_data.json"
    #with open(file_path, "r") as json_file:
    #    CAS_data = json.load(json_file) # 31424
    
    result = []
    filtered_data = []
    for i in range(len(PreparedData)):
        check_CAS = True
        for j in range(len(PreparedData[i]['Precursors'])):
            # Removing non-CAS precursor cases
            if PreparedData[i]['Precursors'][j] in ['LiO','Li2O3','H2B2O3','H2BO3','Na2O3','SiO4','CaCaO3','CaO3',
                                                    'TiO5','TiO3','V2O','CrO2','CrO2.65','MnO3','MnO1.839','Mn2O7',
                                                    'Mn4O3','MnO4','Mn2(C2O4)7','MnC2O4','Fe2O5','Fe2O','Fe2O4',
                                                    'Fe(C2O4)2','Co3O','CoO2','Co3O4.24','NiO2','CuO2','Cu2O3',
                                                    'ZnO3','Sr2O3','Mo2O3','MoO','MoO6','Sb2O4','Ba2O3','BaO3',
                                                    'Pr12O22','Pr4O13','PrO','Tb7O11','Tb7O12','Tb6O11','Tb2O7',
                                                    'TbO2','W2O3','Re2O3','Pb2O3','PbPbO4','PbO3','Am0.07','Am']:
                check_CAS = False
            # Revising non-CAS precursor cases
            elif PreparedData[i]['Precursors'][j] in ['LiCO3','Li3CO3']:
                PreparedData[i]['Precursors'][j] = 'Li2CO3'
            elif PreparedData[i]['Precursors'][j] in ['Li(OH)','Li7OH']:
                PreparedData[i]['Precursors'][j] = 'LiOH'
            elif PreparedData[i]['Precursors'][j] in ['LiCH3COO']:
                PreparedData[i]['Precursors'][j] = 'CH3COOLi'
            elif PreparedData[i]['Precursors'][j] in ['B2O5']:
                PreparedData[i]['Precursors'][j] = 'B2O3'
            elif PreparedData[i]['Precursors'][j] in ['B(OH)3','BO3H3']:
                PreparedData[i]['Precursors'][j] = 'H3BO3'
            elif PreparedData[i]['Precursors'][j] in ['NaCO3']:
                PreparedData[i]['Precursors'][j] = 'Na2CO3'
            elif PreparedData[i]['Precursors'][j] in ['MgO2']:
                PreparedData[i]['Precursors'][j] = 'MgO'
            elif PreparedData[i]['Precursors'][j] in ['(MgCO3)4.Mg(OH)2','(MgCO3)4Mg(OH)2']:
                PreparedData[i]['Precursors'][j] = 'Mg5(CO3)4(OH)2'
            elif PreparedData[i]['Precursors'][j] in ['((Al2O3))','Al2O2']:
                PreparedData[i]['Precursors'][j] = 'Al2O3'
            elif PreparedData[i]['Precursors'][j] in ['Al(OH)2']:
                PreparedData[i]['Precursors'][j] = 'Al(OH)3'
            elif PreparedData[i]['Precursors'][j] in ['AlOOH']:
                PreparedData[i]['Precursors'][j] = 'AlO(OH)'
            elif PreparedData[i]['Precursors'][j] in ['C8H20O4Si']:
                PreparedData[i]['Precursors'][j] = 'Si(OC2H5)4'
            elif PreparedData[i]['Precursors'][j] in ['(NH4)H2PO4','H2NH4PO4','NH4(H2PO4)','(NH4)(H2PO4)','NH4(H2)PO4']:
                PreparedData[i]['Precursors'][j] = 'NH4H2PO4'
            elif PreparedData[i]['Precursors'][j] in ['(NH4)2.HPO4']:
                PreparedData[i]['Precursors'][j] = '(NH4)2HPO4'
            elif PreparedData[i]['Precursors'][j] in ['P4P4']:
                PreparedData[i]['Precursors'][j] = 'P'
            elif PreparedData[i]['Precursors'][j] in ['(NH4)2(HSO4)2']:
                PreparedData[i]['Precursors'][j] = '(NH4)(HSO4)'
            elif PreparedData[i]['Precursors'][j] in ['K2O','K2O3']:
                PreparedData[i]['Precursors'][j] = 'KO2'
            elif PreparedData[i]['Precursors'][j] in ['Ca(CO3)2','Ca2CO3']:
                PreparedData[i]['Precursors'][j] = 'CaCO3'
            elif PreparedData[i]['Precursors'][j] in ['Ca(NO3)']:
                PreparedData[i]['Precursors'][j] = 'Ca(NO3)2'
            elif PreparedData[i]['Precursors'][j] in ['Ti(C4H9O)4','C16H36O4Ti','Ti(OCH2CH2CH2CH3)4']:
                PreparedData[i]['Precursors'][j] = 'Ti(OC4H9)4'
            elif PreparedData[i]['Precursors'][j] in ['C12H28O4Ti']:
                PreparedData[i]['Precursors'][j] = 'Ti(OCH(CH3)2)4'
            elif PreparedData[i]['Precursors'][j] in ['Cr(NO3)6','Cr(NO3)2']:
                PreparedData[i]['Precursors'][j] = 'Cr(NO3)3'
            elif PreparedData[i]['Precursors'][j] in ['Mn2(CO3)7','Mn2CO3']:
                PreparedData[i]['Precursors'][j] = 'MnCO3'
            elif PreparedData[i]['Precursors'][j] in ['Mn(NO3)7']:
                PreparedData[i]['Precursors'][j] = 'Mn(NO3)2'
            elif PreparedData[i]['Precursors'][j] in ['(CH3COO)2Mn','Mn(COOCH3)2']:
                PreparedData[i]['Precursors'][j] = 'Mn(CH3COO)2'
            elif PreparedData[i]['Precursors'][j] in ['MnO(OH)']:
                PreparedData[i]['Precursors'][j] = 'MnOOH'
            elif PreparedData[i]['Precursors'][j] in ['Co(CO3)','Co2(CO3)3']:
                PreparedData[i]['Precursors'][j] = 'CoCO3'
            elif PreparedData[i]['Precursors'][j] in ['Co(OH)3']:
                PreparedData[i]['Precursors'][j] = 'Co(OH)2'
            elif PreparedData[i]['Precursors'][j] in ['Co(NO3)3']:
                PreparedData[i]['Precursors'][j] = 'Co(NO3)2'
            elif PreparedData[i]['Precursors'][j] in ['Co(CH3COO)3','(CH3COO)2Co']:
                PreparedData[i]['Precursors'][j] = 'Co(CH3COO)2'
            elif PreparedData[i]['Precursors'][j] in ['Co2']:
                PreparedData[i]['Precursors'][j] = 'Co'
            elif PreparedData[i]['Precursors'][j] in ['Co2(C2O4)3']:
                PreparedData[i]['Precursors'][j] = 'CoC2O4'
            elif PreparedData[i]['Precursors'][j] in ['Ni2(CO3)3']:
                PreparedData[i]['Precursors'][j] = 'NiCO3'
            elif PreparedData[i]['Precursors'][j] in ['Ni(OH)3']:
                PreparedData[i]['Precursors'][j] = 'Ni(OH)2'
            elif PreparedData[i]['Precursors'][j] in ['Ni(NO3)3']:
                PreparedData[i]['Precursors'][j] = 'Ni(NO3)2'
            elif PreparedData[i]['Precursors'][j] in ['Ni(CH3COO)3']:
                PreparedData[i]['Precursors'][j] = 'Ni(CH3COO)2'
            elif PreparedData[i]['Precursors'][j] in ['Cu(NO3)3']:
                PreparedData[i]['Precursors'][j] = 'Cu(NO3)2'
            elif PreparedData[i]['Precursors'][j] in ['GaO2']:
                PreparedData[i]['Precursors'][j] = 'Ga2O3'
            elif PreparedData[i]['Precursors'][j] in ['Ge2O3']:
                PreparedData[i]['Precursors'][j] = 'GeO2'
            elif PreparedData[i]['Precursors'][j] in ['Sr2CO3','Sr(CO3)2','Sr(CO3)','Sr3CO3']:
                PreparedData[i]['Precursors'][j] = 'SrCO3'
            elif PreparedData[i]['Precursors'][j] in ['Sr(NO3)3']:
                PreparedData[i]['Precursors'][j] = 'Sr(NO3)2'
            elif PreparedData[i]['Precursors'][j] in ['Sr(C2H3O2)2']:
                PreparedData[i]['Precursors'][j] = 'Sr(CH3COO)2'
            elif PreparedData[i]['Precursors'][j] in ['YO1.5','Y3O3']:
                PreparedData[i]['Precursors'][j] = 'Y2O3'
            elif PreparedData[i]['Precursors'][j] in ['Y(NO3)']:
                PreparedData[i]['Precursors'][j] = 'Y(NO3)3'
            elif PreparedData[i]['Precursors'][j] in ['ZrO3','ZrO','Zr2O']:
                PreparedData[i]['Precursors'][j] = 'ZrO2'
            elif PreparedData[i]['Precursors'][j] in ['Nb2O3','Nb2O9','Nb2O6']:
                PreparedData[i]['Precursors'][j] = 'Nb2O5'
            elif PreparedData[i]['Precursors'][j] in ['RhO2']:
                PreparedData[i]['Precursors'][j] = 'Rh2O3'
            elif PreparedData[i]['Precursors'][j] in ['Ag2NO3','Ag(NO3)2']:
                PreparedData[i]['Precursors'][j] = 'AgNO3'
            elif PreparedData[i]['Precursors'][j] in ['TeO3','Te2O']:
                PreparedData[i]['Precursors'][j] = 'TeO2'
            elif PreparedData[i]['Precursors'][j] in ['H6TeO6']:
                PreparedData[i]['Precursors'][j] = 'Te(OH)6'
            elif PreparedData[i]['Precursors'][j] in ['Ba2CO3','Ba3CO3']:
                PreparedData[i]['Precursors'][j] = 'BaCO3'
            elif PreparedData[i]['Precursors'][j] in ['Ba(NO3)']:
                PreparedData[i]['Precursors'][j] = 'Ba(NO3)2'
            elif PreparedData[i]['Precursors'][j] in ['Ba(C2H3O2)2']:
                PreparedData[i]['Precursors'][j] = 'Ba(CH3COO)2'
            elif PreparedData[i]['Precursors'][j] in ['La2CO3','LaCO3']:
                PreparedData[i]['Precursors'][j] = 'La2(CO3)3'
            elif PreparedData[i]['Precursors'][j] in ['LaO3','LaO1.5','La2O5']:
                PreparedData[i]['Precursors'][j] = 'La2O3'
            elif PreparedData[i]['Precursors'][j] in ['La(NO3)2']:
                PreparedData[i]['Precursors'][j] = 'La(NO3)3'
            elif PreparedData[i]['Precursors'][j] in ['CeO','Ce2O3']:
                PreparedData[i]['Precursors'][j] = 'CeO2'
            elif PreparedData[i]['Precursors'][j] in ['Ce(NO3)4','Ce(NO3)']:
                PreparedData[i]['Precursors'][j] = 'Ce(NO3)3'
            elif PreparedData[i]['Precursors'][j] in ['Nd2O5']:
                PreparedData[i]['Precursors'][j] = 'Nd2O3'
            elif PreparedData[i]['Precursors'][j] in ['Sm2O']:
                PreparedData[i]['Precursors'][j] = 'Sm2O3'
            elif PreparedData[i]['Precursors'][j] in ['Sm(NO3)2']:
                PreparedData[i]['Precursors'][j] = 'Sm(NO3)3'
            elif PreparedData[i]['Precursors'][j] in ['EuO']:
                PreparedData[i]['Precursors'][j] = 'Eu2O3'
            elif PreparedData[i]['Precursors'][j] in ['DyO1.5']:
                PreparedData[i]['Precursors'][j] = 'Dy2O3'
            elif PreparedData[i]['Precursors'][j] in ['HoO']:
                PreparedData[i]['Precursors'][j] = 'Ho2O3'
            elif PreparedData[i]['Precursors'][j] in ['Ta2O9','Ta2O3']:
                PreparedData[i]['Precursors'][j] = 'Ta2O5'
            elif PreparedData[i]['Precursors'][j] in ['(NH4)10H2W12O42']:
                PreparedData[i]['Precursors'][j] = '(NH4)10H2(W2O7)6'
            elif PreparedData[i]['Precursors'][j] in ['IrO6']:
                PreparedData[i]['Precursors'][j] = 'IrO2'
            elif PreparedData[i]['Precursors'][j] in ['BiO1.5','Bi3O','BiO3','Bi2O5']:
                PreparedData[i]['Precursors'][j] = 'Bi2O3'
            elif PreparedData[i]['Precursors'][j] in ['(Pu0.93)O2']:
                PreparedData[i]['Precursors'][j] = 'PuO2' 
        if check_CAS:
            result.append(PreparedData[i])
        else:
            filtered_data.append(PreparedData[i])
      
    return result, filtered_data

def select_SingleSourcePrecursors(PreparedData):
    """
    1. One precursor has one Source_element (related to easily accessible, affordable precursors)
    2. Number of Target source element == Number of Precursors
    3. One target has at least one Source_element
    """
    result = []
    filtered_data = []
    for i in range(len(PreparedData)):
        check_tar = True
        tar_source_elem, _ = get_SourceElem(PreparedData[i]['Target'], comp_type='Target')
        for j in range(len(PreparedData[i]['Target'])):
            t_source_elem, _ = get_SourceElem([PreparedData[i]['Target'][j]], comp_type='Target')
            source_count = len(t_source_elem)
            if source_count == 0:
                check_tar = False
                break

        single_source_check = True
        total_pre_source_count = 0
        pre_source_elem = []
        for j in range(len(PreparedData[i]['Precursors'])):
            p_source_elem, _ = get_SourceElem([PreparedData[i]['Precursors'][j]], comp_type='Precursor')
            source_count = len(p_source_elem)
            total_pre_source_count += 1
            if source_count != 1:
                single_source_check = False
                break
            else:
                pre_source_elem.append(p_source_elem[0])
        if single_source_check and check_tar and (len(tar_source_elem) == total_pre_source_count) and (set(tar_source_elem)==set(pre_source_elem)):
            result.append(PreparedData[i])
        else:
            filtered_data.append(PreparedData[i])
    return result, filtered_data

def remove_Duplicate(data):
    result = []
    tar_pre_dict = {}
    count = 0
    for i in range(len(data)):
        tar_list = data[i]['Target']
        tar_list.sort()

        pre_list = data[i]['Precursors']
        pre_list.sort()

        tar = str(tar_list)
        pre = str(pre_list)
        if tar not in tar_pre_dict:
            tar_pre_dict[tar] = []
            tar_pre_dict[tar].append(pre)
            count += 1
            result.append(data[i])
        else:
            if pre not in tar_pre_dict[tar]:
                tar_pre_dict[tar].append(pre)
                count += 1
                result.append(data[i])
    return result

def remove_Duplicate_forTPO(data):
    result = []
    tar_pre_dict = {}
    tar_pre_idx_dict = {}
    count = 0
    for i in range(len(data)):
        tar_list = data[i]['Target']
        tar_list.sort()

        pre_list = data[i]['Precursors']
        pre_list.sort()

        tar = str(tar_list)
        pre = str(pre_list)
        if tar == pre:
            pass
        else:
            tar_pre = tar+"--"+pre
            if tar not in tar_pre_dict:
                tar_pre_dict[tar] = []
                tar_pre_dict[tar].append(pre)
                count += 1
                data[i]['tarpre'] = tar_pre
                tar_pre_idx_dict[tar_pre] = i
                result.append(data[i])
            else:
                if pre not in tar_pre_dict[tar]:
                    tar_pre_dict[tar].append(pre)
                    count += 1
                    data[i]['tarpre'] = tar_pre
                    tar_pre_idx_dict[tar_pre] = i
                    result.append(data[i])
                else:
                    for j in range(len(data[i]['Operation'])):
                        data[tar_pre_idx_dict[tar_pre]]['Operation'].append(data[i]['Operation'][j])
    for i in range(len(result)):
        del result[i]['tarpre']

    return result

def MultiTemp_preprocessing(data, histshow):
    result = []
    filtered_data = []
    temp_std_hist = []
    for i in range(len(data)):
        temp_set = []
        for j in range(len(data[i]['Operation'])):
            temp = float(data[i]['Operation'][j][1])
            temp_set.append(temp)

        if len(temp_set) == 1:  # One-step heating
            data[i]['Operation'] = [round(temp,1)]
            result.append(data[i])
        elif len(temp_set) > 1:
            temp_mean = torch.mean(torch.tensor(temp_set)).item()
            temp_std = torch.std(torch.tensor(temp_set), unbiased=False).item()
            temp_std_hist.append(temp_std)
            if 0<= temp_std <= 200:
                data[i]['Operation'] = [round(temp_mean,1)]
                result.append(data[i])
            else:
                filtered_data.append(data[i])
        else:
            filtered_data.append(data[i])

    if histshow:
        plt.hist(temp_std_hist, bins=30)
        plt.show()

    return result, filtered_data

def get_AnionPart(composition, source_elem, ExceptionMode=False, TargetTypeMode=False):
    comp_dict = Composition(composition).get_el_amt_dict()
    ca_count = 0
    an_count = 0
    anion = ""
    for elem, stoi in comp_dict.items():
        if TargetTypeMode:
            if str(elem) in inorg_elem:
            #if str(elem) in source_elem:
                ca_count += 1
            else:
                an_count += 1
                anion += str(elem)+str(stoi)
        else:
            if str(elem) in source_elem:
                ca_count += 1
            else:
                an_count += 1
                anion += str(elem)+str(stoi)
    if ca_count == 0:
        if ExceptionMode:
            pass
        else:
            raise NotImplementedError('No source elem', composition)

    if anion != "":
        anion = str(Composition(anion).get_integer_formula_and_factor()[0])
    return anion

def find_TotalAnionFramework(data, including_elem='All', domain='Precursors', ExceptionMode=False, TargetTypeMode=False):
    pre_anion_part = {}
    for i in range(len(data)):
        # Finding Precursors Anion Framework
        if domain == 'Precursors':
            for j in range(len(data[i][domain])):
                p_source_elem, _ = get_SourceElem([data[i][domain][j]], comp_type='Precursor')
                if including_elem == 'All':
                    pre_anion = get_AnionPart(data[i][domain][j], p_source_elem, ExceptionMode, TargetTypeMode)
                    if pre_anion in pre_anion_part:
                        pre_anion_part[pre_anion] += 1
                    else:
                        pre_anion_part[pre_anion] = 1
                else:
                    if including_elem in Composition(data[i][domain][j]):
                        pre_anion = get_AnionPart(data[i][domain][j], p_source_elem, ExceptionMode, TargetTypeMode)
                        if pre_anion in pre_anion_part:
                            pre_anion_part[pre_anion] += 1
                        else:
                            pre_anion_part[pre_anion] = 1
        elif domain == 'Target':
            if len(data[i][domain]) > 1:
                pre_anion = 'Composite'
                if pre_anion in pre_anion_part:
                    pre_anion_part[pre_anion] += 1
                else:
                    pre_anion_part[pre_anion] = 1
            else:
                for j in range(len(data[i][domain])):
                    t_source_elem, _ = get_SourceElem([data[i][domain][j]], comp_type='Precursor')
                    pre_anion = get_AnionPart(data[i][domain][j], t_source_elem, ExceptionMode, TargetTypeMode)
                    if pre_anion in pre_anion_part:
                        pre_anion_part[pre_anion] += 1
                    else:
                        pre_anion_part[pre_anion] = 1
    return pre_anion_part

def most_frequent_stoi_dict_from_source_template(data, pre_anion_part):
    stoi_ll_dict = {}
    for s_elem in elem_library:
        for template in pre_anion_part:
            stoi_ll_dict[s_elem+template] = {}
    for i in tqdm(range(len(data))):
        for j in range(len(data[i]['Precursors'])):
            p_source_elem, _ = get_SourceElem([data[i]['Precursors'][j]])
            template = get_AnionPart(data[i]['Precursors'][j], p_source_elem)
            if len(p_source_elem) == 1:
                if data[i]['Precursors'][j] not in stoi_ll_dict[p_source_elem[0]+template]:
                    stoi_ll_dict[p_source_elem[0]+template][data[i]['Precursors'][j]] = 1
                else:
                    stoi_ll_dict[p_source_elem[0]+template][data[i]['Precursors'][j]] += 1
            else:
                raise NotImplementedError("No single source element precursors")

    most_count = 0
    minor_count = 0
    stoi_dict = {}
    for s_elem in elem_library:
        for template in pre_anion_part:
            stoi_dict[s_elem+template] = []
            if len(stoi_ll_dict[s_elem+template]) != 0:
                di = stoi_ll_dict[s_elem+template]
                most_frequent = [k for k,v in di.items() if max(di.values()) == v]
                if len(most_frequent) == 1:
                    stoi_dict[s_elem+template] = [most_frequent[0]]
                else:
                    #print(stoi_ll_dict[s_elem+template])
                    stoi_dict[s_elem+template] = [most_frequent[0]]
                for k,v in di.items():
                    minor_count += v
                most_count += max(di.values())
                minor_count -= max(di.values())

    return stoi_dict, most_count, minor_count

def stoi_count_dict_from_source_template(data, pre_anion_part):
    stoi_ll_dict = {}
    for s_elem in elem_library:
        for template in pre_anion_part:
            stoi_ll_dict[s_elem+template] = {}
    for i in tqdm(range(len(data))):
        for j in range(len(data[i]['Precursors'])):
            p_source_elem, _ = get_SourceElem([data[i]['Precursors'][j]])
            template = get_AnionPart(data[i]['Precursors'][j], p_source_elem)
            if len(p_source_elem) == 1:
                if data[i]['Precursors'][j] not in stoi_ll_dict[p_source_elem[0]+template]:
                    stoi_ll_dict[p_source_elem[0]+template][data[i]['Precursors'][j]] = 1
                else:
                    stoi_ll_dict[p_source_elem[0]+template][data[i]['Precursors'][j]] += 1
            else:
                raise NotImplementedError("No single source element precursors")
    return stoi_ll_dict


elem_library            = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',
                           'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe',
                           'Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr',
                           'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
                           'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm',
                           'Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W',
                           'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
                           'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf',
                           'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds',
                           'Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

alkali_metal            = ['Li','Na','K','Rb','Cs']
alkaline_earth_metal    = ['Be','Mg','Ca','Sr','Ba']
transition_metal        = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                           'Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','Hf',
                           'Ta','W','Re','Os','Ir','Pt','Au','Hg']
lanthanide_elem         = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
actinide_elem           = ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
post_transition_metal   = ['Al','Ga','In','Sn','Tl','Pb','Bi']
metalloid               = ['B','Si','Ge','As','Sb','Te']
non_metal               = ['H','C','N','O','F','P','S','Cl','Se','Br','I']
noble_gas               = ['He','Ne','Ar','Kr','Xe']
artificial_elem         = ['Tc','Pm','Po','At','Rn','Fr','Ra','Rf','Db','Sg','Bh',
                           'Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

essen_elem = alkali_metal + alkaline_earth_metal + transition_metal \
             + lanthanide_elem + actinide_elem + post_transition_metal + metalloid + ['P','Se','S']

inorg_elem = alkali_metal + alkaline_earth_metal + transition_metal \
             + lanthanide_elem + actinide_elem + post_transition_metal + metalloid


def show_data_coverage(data_TP):
    print("Showing data coverage plot ... \n")
    whole_data_TP = remove_Duplicate(data_TP)
    print(len(whole_data_TP))     # 14365

    tar_anion_part = find_TotalAnionFramework(whole_data_TP, including_elem='All', domain='Target', ExceptionMode=True, TargetTypeMode=True)
    #sorted_tar_anion_part = sorted(tar_anion_part.items(), key=operator.itemgetter(1))
    #for ap in sorted_tar_anion_part:
    #    print(ap)

    whole_oxide_num = 0
    whole_alloy_num = 0
    whole_halide_num = 0
    whole_oxyhalide_num = 0
    whole_carbide_num = 0
    whole_nitride_num = 0
    whole_composite_num = 0
    whole_hydride_num = 0
    whole_phosphate_num = 0
    whole_pyrophosphate_num = 0
    whole_selenide_num = 0
    whole_sulfide_num = 0
    whole_num = 0

    low_coverage_case = []
    for i in range(len(whole_data_TP)):
        if len(whole_data_TP[i]['Target']) > 1:
            whole_composite_num += 1
        else:
            for j in range(len(whole_data_TP[i]['Target'])):
                t_source_elem, _ = get_SourceElem([whole_data_TP[i]['Target'][j]], comp_type='Precursor')
                anion = get_AnionPart(whole_data_TP[i]['Target'][j], t_source_elem, ExceptionMode=True, TargetTypeMode=True)
                if anion == 'O2':
                    whole_oxide_num += 1
                elif anion == 'Composite':
                    whole_composite_num += 1
                elif anion == '':
                    whole_alloy_num += 1
                elif anion == 'PO4':
                    whole_phosphate_num += 1
                elif anion == 'P2O7':
                    whole_pyrophosphate_num += 1
                elif anion in ['F2', 'Cl2', 'Br', 'I']:
                    whole_halide_num += 1
                    #low_coverage_case.append(whole_data_TP[i])
                elif set(list(Composition(anion).get_el_amt_dict().keys())) in [set(['F','O']), set(['Cl','O']), set(['Br','O']), set(['I','O'])]:
                    whole_oxyhalide_num += 1
                    low_coverage_case.append(whole_data_TP[i])
                elif anion == 'C':
                    whole_carbide_num += 1
                    low_coverage_case.append(whole_data_TP[i])
                elif anion == 'N2':
                    whole_nitride_num += 1
                    #low_coverage_case.append(whole_data_TP[i])
                elif anion == 'H2':
                    whole_hydride_num += 1
                    #low_coverage_case.append(whole_data_TP[i])
                elif anion == 'Se':
                    whole_selenide_num += 1
                elif anion == 'S':
                    whole_sulfide_num += 1
        whole_num += 1

    low_coverage_case, f = select_SingleSourcePrecursors(low_coverage_case)
    data_TP, f = select_SingleSourcePrecursors(whole_data_TP)

    oxide_num = 0
    alloy_num = 0
    halide_num = 0
    oxyhalide_num = 0
    carbide_num = 0
    nitride_num = 0
    composite_num = 0
    hydride_num = 0
    phosphate_num = 0
    pyrophosphate_num = 0
    selenide_num = 0
    sulfide_num = 0
    num = 0
    for i in range(len(data_TP)):
        if len(data_TP[i]['Target']) > 1:
            composite_num += 1
        else:
            for j in range(len(data_TP[i]['Target'])):
                t_source_elem, _ = get_SourceElem([data_TP[i]['Target'][j]], comp_type='Precursor')
                anion = get_AnionPart(data_TP[i]['Target'][j], t_source_elem, ExceptionMode=True, TargetTypeMode=True)
                if anion == 'O2':
                    oxide_num += 1
                elif anion == 'Composite':
                    composite_num += 1
                elif anion == '':
                    alloy_num += 1
                elif anion == 'PO4':
                    phosphate_num += 1
                elif anion == 'P2O7':
                    pyrophosphate_num += 1
                elif anion in ['F2', 'Cl2', 'Br', 'I']:
                    halide_num += 1
                elif set(list(Composition(anion).get_el_amt_dict().keys())) in [set(['F','O']), set(['Cl','O']), set(['Br','O']), set(['I','O'])]:
                    oxyhalide_num += 1
                elif anion == 'C':
                    carbide_num += 1
                elif anion == 'N2':
                    nitride_num += 1
                elif anion == 'Se':
                    selenide_num += 1
                elif anion == 'S':
                    sulfide_num += 1
        num += 1

    print('oxide : ',oxide_num,'/',whole_oxide_num,'\n'+
          'composite : ',composite_num,'/',whole_composite_num,'\n'+
          'alloy : ',alloy_num,'/',whole_alloy_num,'\n'+
          'phosphate : ',phosphate_num,'/',whole_phosphate_num,'\n'+
          'oxyhalide : ',oxyhalide_num,'/',whole_oxyhalide_num,'\n'+
          'sulfide : ',sulfide_num,'/',whole_sulfide_num,'\n'+
          'selenide : ',selenide_num,'/',whole_selenide_num,'\n'+
          'halide : ',halide_num,'/',whole_halide_num,'\n'+
          'pyrophosphate : ',pyrophosphate_num,'/',whole_pyrophosphate_num,'\n'+
          'carbide : ',carbide_num,'/',whole_carbide_num,'\n'+
          'nitride : ',nitride_num,'/',whole_nitride_num,'\n'+
          'hydride : ',hydride_num,'/',whole_hydride_num,'\n'+
          'total : ',num,'/',whole_num)

    x = np.arange(13)
    #x = np.arange(10)
    target_type = ['Oxide', 'Composite', 'Alloy', 'Phosphate', 'Oxyhalide', 'Sulfide', 'Selenide', 'Halide', 'Pyrophosphate', 
                   'Carbide', 'Nitride', 'Hydride',
                   'Total']
    #target_type = ['Oxide']*13
    coverage_ratio = [oxide_num/whole_oxide_num,
                      composite_num/whole_composite_num,
                      alloy_num/whole_alloy_num,
                      phosphate_num/whole_phosphate_num,
                      oxyhalide_num/whole_oxyhalide_num,
                      sulfide_num/whole_sulfide_num,
                      selenide_num/whole_selenide_num,
                      halide_num/whole_halide_num,
                      pyrophosphate_num/whole_pyrophosphate_num,
                      carbide_num/whole_carbide_num,
                      nitride_num/whole_nitride_num,
                      hydride_num/whole_hydride_num,
                      num/whole_num]
    coverage_ratio = np.array(coverage_ratio)*100
    plt.figure(figsize=(22,8)).patch.set_alpha(0)
    plt.plot(x, coverage_ratio, color='red', marker='o', markersize=20, linewidth=4, linestyle='--')
    for i in range(len(coverage_ratio)):
        height = coverage_ratio[i]
        plt.text(x[i], height+4, '%.1f' %height, ha='center', va='bottom', size = 27, color='red')
    plt.gca().spines['left'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0)
    plt.gca().tick_params(width=2.5, length=8)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().patch.set_alpha(0)
    plt.ylabel('Reaction coverage (%)', size=40, labelpad=15, color='red')
    plt.xticks(x, target_type, rotation=45, size=30)
    plt.yticks(size=30, color='red')
    plt.ylim([-0.1*100,1.1*100])
    plt.xlim([-0.7,12.7])
    plt.show()
    
    y = np.arange(5)
    type_count = [whole_oxide_num, whole_composite_num, whole_alloy_num, whole_phosphate_num,
                  whole_oxyhalide_num, whole_sulfide_num, whole_selenide_num, whole_pyrophosphate_num,
                  whole_halide_num, whole_carbide_num, whole_nitride_num, whole_hydride_num, whole_num]
    type_log_count = []
    for cc in type_count:
        type_log_count.append(math.log(cc, 10))
    plt.figure(figsize=(22,8))
    plt.bar(x, type_log_count, width=0.6 ,color=['dimgrey']*12+['black'])
    plt.gca().spines['left'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().tick_params(width=2.5, length=8)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel('Inorganic target types', size=35, labelpad=15)
    plt.ylabel('Counts ($10^{n}$)', size=40, labelpad=15)
    plt.xticks(x, target_type, rotation=45, size=30)
    plt.yticks(y, 10**y, size=30)
    plt.xlim([-0.7,12.7])
    plt.show()
    
def show_AnionTemplate_population(pre_anion_part):
    sorted_pre_anion_part = sorted(pre_anion_part.items(), key=operator.itemgetter(1))
    #for ap in sorted_pre_anion_part:
    #    print(ap)
    sorted_pre_anion_part.reverse()
    top_frequent_anion = sorted_pre_anion_part[:15]
    
    x = np.arange(len(top_frequent_anion))
    y = np.arange(5)
    pre_anion_type = []
    pre_anion_count = []
    for an in top_frequent_anion:
        if an[0] == '':
            pre_anion_type.append('Pure metal')
        else:
            pre_anion_type.append(an[0])
        pre_anion_count.append(math.log(an[1], 10))
        
    plt.figure(figsize=(22,8))
    plt.bar(x, pre_anion_count, width=0.6 ,color=['tab:blue']*len(top_frequent_anion))
    plt.gca().spines['left'].set_linewidth(2.5)
    plt.gca().spines['right'].set_linewidth(2.5)
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().tick_params(width=2.5, length=8)
    plt.xlabel('Precursor anion template types', size=35, labelpad=15)
    plt.ylabel('Counts ($10^{n}$)', size=40, labelpad=15)
    plt.xticks(x, pre_anion_type, rotation=45, size=30)
    plt.yticks(y, 10**y, size=30)
    plt.show()


# Prepare data
def DataPreparation():
    # Text-mined dataset load
    with open("dataset/solid-state_dataset_20200713.json", 'r', encoding='utf-8-sig') as json_file:
        data = json.load(json_file) # 31782
    ordered_syn_elem_library = get_ordered_syn_elem_library(data)   # target_elem in elem_library

    data_TPO, f, data_TP = select_CorrectlyParsedSynData(data, ordered_syn_elem_library)

    print(len(data_TP))     # 25873
    data_TP, f = select_MassConservation(data_TP)
    print(len(data_TP))     # 22837
    data_TP, f = select_CommerciallyViable(data_TP)
    print(len(data_TP))     # 22705

    show_data_coverage(data_TP)

    data_TP, f = select_SingleSourcePrecursors(data_TP)
    print(len(data_TP))     # 21085
    data_TP = remove_Duplicate(data_TP)
    print(len(data_TP))     # 13477

    print(len(data_TPO))    # 20764
    data_TPO, f = select_MassConservation(data_TPO)
    print(len(data_TPO))    # 18271
    data_TPO, f = select_CommerciallyViable(data_TPO)
    print(len(data_TPO))    # 18164
    data_TPO, f = select_SingleSourcePrecursors(data_TPO)
    print(len(data_TPO))    # 17102
    data_TPO = remove_Duplicate_forTPO(data_TPO)
    print(len(data_TPO))    # 11062
    data_TPO, f = MultiTemp_preprocessing(data_TPO, histshow=False)
    print(len(data_TPO))    # 9163

    return data_TP, data_TPO


if __name__ == "__main__":
    data_TP, data_TPO = DataPreparation()
    
    pre_anion_part = find_TotalAnionFramework(data_TP, including_elem='All', domain='Precursors', ExceptionMode=True)
    show_AnionTemplate_population(pre_anion_part)
    
    stoi_ll_dict = stoi_count_dict_from_source_template(data_TP, pre_anion_part)
    stoi_dict, _, _ = most_frequent_stoi_dict_from_source_template(data_TP, pre_anion_part)
    
    print(len(data_TP))     # 13477
    print(len(data_TPO))    # 9163
    
    #get_tar_elementdist(data_TP)
    file_path = "./dataset/pre_anion_part.json"
    with open(file_path, 'w') as outfile:
        json.dump(pre_anion_part, outfile, indent=4)
    
    file_path = "./dataset/stoi_ll_dict.json"
    with open(file_path, 'w') as outfile:
        json.dump(stoi_ll_dict, outfile, indent=4)
        
    file_path = "./dataset/stoi_dict.json"
    with open(file_path, 'w') as outfile:
        json.dump(stoi_dict, outfile, indent=4)

    file_path = "./dataset/InorgSyn_dataset_TP.json"
    with open(file_path, 'w') as outfile:
        json.dump(data_TP, outfile, indent=4)

    file_path = "./dataset/InorgSyn_dataset_TPO.json"
    with open(file_path, 'w') as outfile:
        json.dump(data_TPO, outfile, indent=4)

    """
    # Obtaining the pubdate information using scopus_api request
    your_scopus_apikey = input("Type your scopus apikey to make timesplit dataset with pubdate information = ")
    data_TP2 = doi2pubdate(data_TP, your_scopus_apikey)  # (Time-consuming process) If you don't want, skip this process

    file_path = "./dataset/InorgSyn_dataset_TP2.json"
    with open(file_path, 'w') as outfile:
        json.dump(data_TP2, outfile, indent=4)
    """