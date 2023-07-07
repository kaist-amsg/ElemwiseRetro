# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:28:54 2020

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
## determine elemental position
#x = np.arange(18,dtype=np.float)
#y = -np.arange(9,dtype=np.float) + 8.0
#y[-2:] += -0.2
#from mendeleev import elements
#elems = elements.get_all_elements()
#exy = {}
#for e in elems:
#    if e.block != 'f':
#        exy[e.symbol] = [x[e.group_id-1],y[e.period-1]]
#    else:
#        if e.period == 6:
#            exy[e.symbol] = [x[e.atomic_number-55],y[e.period+1]]
#        if e.period == 7:
#            exy[e.symbol] = [x[e.atomic_number-87],y[e.period+1]]
exy = {'Ac': [2.0, 2.0], 'Ag': [10.0, 4.0], 'Al': [12.0, 6.0], 'Am': [8.0, -0.2], 'Ar': [17.0, 6.0], 'As': [14.0, 5.0], 'At': [16.0, 3.0], 'Au': [10.0, 3.0], 'B': [12.0, 7.0], 'Ba': [1.0, 3.0], 'Be': [1.0, 7.0], 'Bh': [6.0, 2.0], 'Bi': [14.0, 3.0], 'Bk': [10.0, -0.2], 'Br': [16.0, 5.0], 'C': [13.0, 7.0], 'Ca': [1.0, 5.0], 'Cd': [11.0, 4.0], 'Ce': [3.0, 0.8], 'Cf': [11.0, -0.2], 'Cl': [16.0, 6.0], 'Cm': [9.0, -0.2], 'Cn': [11.0, 2.0], 'Co': [8.0, 5.0], 'Cr': [5.0, 5.0], 'Cs': [0.0, 3.0], 'Cu': [10.0, 5.0], 'Db': [4.0, 2.0], 'Ds': [9.0, 2.0], 'Dy': [11.0, 0.8], 'Er': [13.0, 0.8], 'Es': [12.0, -0.2], 'Eu': [8.0, 0.8], 'F': [16.0, 7.0], 'Fe': [7.0, 5.0], 'Fl': [13.0, 2.0], 'Fm': [13.0, -0.2], 'Fr': [0.0, 2.0], 'Ga': [12.0, 5.0], 'Gd': [9.0, 0.8], 'Ge': [13.0, 5.0], 'H': [0.0, 8.0], 'He': [17.0, 8.0], 'Hf': [3.0, 3.0], 'Hg': [11.0, 3.0], 'Ho': [12.0, 0.8], 'Hs': [7.0, 2.0], 'I': [16.0, 4.0], 'In': [12.0, 4.0], 'Ir': [8.0, 3.0], 'K': [0.0, 5.0], 'Kr': [17.0, 5.0], 'La': [2.0, 3.0], 'Li': [0.0, 7.0], 'Lr': [16.0, -0.2], 'Lu': [16.0, 0.8], 'Lv': [15.0, 2.0], 'Mc': [14.0, 2.0], 'Md': [14.0, -0.2], 'Mg': [1.0, 6.0], 'Mn': [6.0, 5.0], 'Mo': [5.0, 4.0], 'Mt': [8.0, 2.0], 'N': [14.0, 7.0], 'Na': [0.0, 6.0], 'Nb': [4.0, 4.0], 'Nd': [5.0, 0.8], 'Ne': [17.0, 7.0], 'Nh': [12.0, 2.0], 'Ni': [9.0, 5.0], 'No': [15.0, -0.2], 'Np': [6.0, -0.2], 'O': [15.0, 7.0], 'Og': [17.0, 2.0], 'Os': [7.0, 3.0], 'P': [14.0, 6.0], 'Pa': [4.0, -0.2], 'Pb': [13.0, 3.0], 'Pd': [9.0, 4.0], 'Pm': [6.0, 0.8], 'Po': [15.0, 3.0], 'Pr': [4.0, 0.8], 'Pt': [9.0, 3.0], 'Pu': [7.0, -0.2], 'Ra': [1.0, 2.0], 'Rb': [0.0, 4.0], 'Re': [6.0, 3.0], 'Rf': [3.0, 2.0], 'Rg': [10.0, 2.0], 'Rh': [8.0, 4.0], 'Rn': [17.0, 3.0], 'Ru': [7.0, 4.0], 'S': [15.0, 6.0], 'Sb': [14.0, 4.0], 'Sc': [2.0, 5.0], 'Se': [15.0, 5.0], 'Sg': [5.0, 2.0], 'Si': [13.0, 6.0], 'Sm': [7.0, 0.8], 'Sn': [13.0, 4.0], 'Sr': [1.0, 4.0], 'Ta': [4.0, 3.0], 'Tb': [10.0, 0.8], 'Tc': [6.0, 4.0], 'Te': [15.0, 4.0], 'Th': [3.0, -0.2], 'Ti': [3.0, 5.0], 'Tl': [12.0, 3.0], 'Tm': [14.0, 0.8], 'Ts': [16.0, 2.0], 'U': [5.0, -0.2], 'V': [4.0, 5.0], 'W': [5.0, 3.0], 'Xe': [17.0, 4.0], 'Y': [2.0, 4.0], 'Yb': [15.0, 0.8], 'Zn': [11.0, 5.0], 'Zr': [3.0, 4.0]}
cs = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
etypes = {'nonmetal':[1,6,7,8,9,15,16,17,34,35,53],\
          'metalloid':[5,14,32,33,51,52],\
          'noble gas':[2,10,18,36,54,86],\
          'post-transition metal':[13, 30, 31, 48, 49, 50, 80, 81, 82, 83, 84, 85],\
          'unknown':[109, 110, 111, 112, 113, 114, 115, 116, 117, 118],\
          'alkali metal':[3, 11, 19, 37, 55, 87],\
          'alkaline earth metal':[4, 12, 20, 38, 56, 88],\
          'transition metal':list(range(21,30))+list(range(39,48))+list(range(72,80))+list(range(104,109)),\
          'lathanide':list(range(57,72)),\
          'actinide':list(range(89,104))}
for e in etypes:
    etypes[e] = [cs[i] for i in etypes[e]]
    
typecolors = {'nonmetal':[0.94117647, 1., 0.56078431, 1.],\
          'metalloid':[0.8, 0.8, 0.6, 1. ],\
          'noble gas':[0.75294118, 1.        , 1.        , 1.        ],\
          'post-transition metal':[0.8,0.8,0.8,1.],\
          'unknown':[0.90980392, 0.90980392, 0.90980392, 1.        ],\
          'alkali metal':[1. , 0.4, 0.4, 1. ],\
          'alkaline earth metal':[1.        , 0.87058824, 0.67843137, 1.        ],\
          'transition metal':[1.        , 0.75294118, 0.75294118, 1.        ],\
          'lathanide':[1.        , 0.74901961, 1.        , 1.        ],\
          'actinide':[1.        , 0.61176471, 0.81568627, 1.        ],}
ecolors = {}
for t in etypes:
    for s in etypes[t]:
        ecolors[s] = typecolors[t]

es = []
xyz = []
cs = []
for e in exy:
    es.append(e)
    xyz.append(exy[e]+[0])
    cs.append(ecolors[e])
xyz = np.array(xyz)

# %matplotlib auto
# %matplotlib inline
def PlotPTable(Zs):
    dz=np.zeros(xyz.shape[0])
    for e in Zs:
        dz[es.index(e)] = Zs[e]
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(projection='3d')
    plt.subplots_adjust(left=0,bottom=0,top=1,right=1)
    ax.bar3d(xyz[:,0],xyz[:,1],xyz[:,2],np.ones(xyz.shape[0])*0.6,\
             np.ones(xyz.shape[0])*0.6,dz,color=cs,\
             shade=False,edgecolor='k')
    ax.set_xlim3d(0,18)
    ax.set_ylim3d(0,9)
    ax.set_axis_off()
    #ax.set_zlim3d(0,10)
    for x,y,z,e in zip(xyz[:,0],xyz[:,1],dz,es):
        ax.text(x+0.3,y+0.3,z,e, horizontalalignment='center', verticalalignment='center',size=16)
    ax.view_init(elev=80,azim=280)
    ax.set_box_aspect([2,1,1])

from matplotlib import cm
def PlotPTable2(Zs):
    dz=np.zeros(xyz.shape[0])
    for e in Zs:
        dz[es.index(e)] = Zs[e]
    ccs = []
    for z in dz:
        ccs.append(cm.Greens(z/np.max(dz)))
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0,bottom=0,top=1,right=1)
    im = ax.scatter(xyz[:,0]+0.25,xyz[:,1]+0.25,s=1500,marker='s',c=dz,edgecolor='k',cmap='viridis')
    ax.set_xlim(-2,20)
    ax.set_ylim(-2,11)
    ax.set_axis_off()
    #ax.set_zlim3d(0,10)
    for x,y,e in zip(xyz[:,0],xyz[:,1],es):
        ax.text(x+0.3,y+0.3,e, horizontalalignment='center', verticalalignment='center',size=20,c='w')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.25, 0.8, 0.4, 0.05])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=20)