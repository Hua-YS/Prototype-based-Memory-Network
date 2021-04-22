import os
import sys
sys.path.append(os.path.abspath('./utils'))
from mai import *
from cnn import *
import numpy as np

# ***************************** path *******************************
backbone = 'resnet50' # CNN backbone (resnet50, vggnet16, inceptionv3, nasnet)
dataset = 'ucm_mu' # creating memory for ucm2mai or aid2mai (ucm_mu, aid_mu)
weight_path = 'weights/ucm2mai/'+backbone+'-ucm.h5'
savename = 'memory_'+backbone # path to save memory matrix

# ************************ configurations **************************
feat_dim = sel_dim(backbone) 
patch_size = 224 # image patch
nb_classes = 20 if dataset=='aid_mu' else 16
gpu_config(0, 0.3) # gpu id, memory consumption

# ************************ initialize model ************************
model = cnn(backbone, nb_classes=nb_classes, patch_size=patch_size, activation='sigmoid', pretrain='imagenet', trainable=False, isbackbone=True)

# ************************ data preparation ************************
X, y, _, _ = load_data(dataset, patch_size, True, 1)

# ************************* training *******************************
model.load_weights(weight_path, by_name=True) 
X_feat = model.predict(X) # extract features of single-scene images

mean_feat = np.zeros((nb_classes, feat_dim)) # creating an empty memory
print('nb_class:', nb_classes)
for i in range(nb_classes):
    idx_pos = np.where(y[:, i]==1)[0]
    mean_feat[i, :] = np.mean(X_feat[idx_pos, :], 0)
    
sio.savemat(savename+'xx.mat', {'mean_feat': mean_feat})

