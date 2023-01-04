import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models import AlexNet3D_Dropout_Regression 
"""Pulling models.py from current directory
At some point a requirement for input_size was added to 
AlexNet3D_Dropout_Regression within DCAN version. Unitl there is a more
elegant solution will use models.py from commit #9511719"""
import sys
import utils as ut
"""Modifying utils.py, specifically the load_net_weights2 function

"""
import torch
from torch.autograd import Variable

###### Motion QC Value REGRESSION ##########

# Parameter list
scorename = 'rating' # Motion QC Regression
mt = 'AlexNet3D_Dropout_Regression'
es_pat = 40
iter_ = 0 # First crossvalidation repetition
lr = 0.001
nc = 1
cmp = 'winter'
ssd = './SampleSplits_Age/'
tr_smp_sizes = [1525]
nReps = 20
mode = 'te'

# Specify model location
ml = 'motion_qc_model_18.pt'

# Read model
net = AlexNet3D_Dropout_Regression()
net = ut.load_net_weights2(net, ml)

# Read labels
labels = pd.read_csv('bcp_and_elabe_qc_train_space-infant_with_init_flag_unique.csv')[scorename].values

# Read data
df_te = ut.readFrames(iter_,tr_smp_sizes,nReps,mode,ssd)
X_te, y_te = ut.read_X_y_5D(df_te,scorename)     
X_te = Variable(torch.from_numpy(X_te))

# Forward Pass (Generate Embeddings)
embs = ut.forward_pass_embeddings(X_te,net,'reg')

# Project Embeddings
X_embs = TSNE(n_components=2, perplexity=100, learning_rate=300, random_state=1).fit_transform(embs)

# Plot Spectra
plt.figure(num=1, figsize=(6, 4), dpi=500, facecolor='w', edgecolor='w')
plt.scatter(X_embs[:,0], X_embs[:, 1], c=labels, s=2, cmap=cmp)
plt.axis('tight')
plt.yticks([])
plt.xticks([])
#plt.legend(loc=9, bbox_to_anchor=(0.5, 0.4), ncol=2, shadow=False, fontsize = 10)
cbar = plt.colorbar(orientation='horizontal', fraction = 0.05)
cbar.set_label('Age')
#plt.show()
plt.savefig('./Figures/age_reg_projection', dpi = 300)
plt.clf()
