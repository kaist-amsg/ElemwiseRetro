# ElemwiseRetro
Implementation of ElemwiseRetro developed by Prof. Yousung Jung group at Seoul National University

(contact: yousung@gmail.com)

( Predicting Synthesis Recipes of Inorganic Crystal Materials using Elementwise Template Formulation )

DOI : 10.1039/D3SC03538G


# Developer
Seongmin Kim (seongminkim0215@gmail.com)


# Python Dependencies
* Python (version == 3.8.13)
* Numpy (version == 1.22.3)
* PyTorch (version == 1.11.0)
* Pymatgen (version == 2022.9.21)


# How to use
First, take zip files from https://zenodo.org/record/8123145 (from Ceder's group, textmined inorganic synthesis dataset),
put them in "./dataset" folder, and execute the below codes

--------------------------------------------------------------------------

Data.py            ; Preprocessing the data

Train_P.py         ; Target -> Precursors prdicting model training and Save (ElemwiseRetro)

Train_T.py         ; Target + Precursors -> Temperature predicting model training and Save

baseline_Model.py	 ; Template popularity based baseline model and Save

Test_TP.py	       ; Load trained models and show the results

