
# coding: utf-8

# In[1]:
'''
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

Collate the results to find the most popular voxlets
'''
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/real_data/"))
import yaml
from oisin_house import real_data_paths as paths
import numpy as np

test_type = 'different_data_split'

model_names = ['Floating voxlet model', 'Tall, fixed voxlet model']
accumulators = {}

for model_idx in [0, 1]:

    all_counts = []

    for sequence in paths.test_data:

        fpath = paths.voxlet_prediction_folderpath %             (test_type, sequence['name'])
        count_path = fpath + '/voxlet_count_%02d.txt' % model_idx

        voxlet_counts = np.genfromtxt(count_path, delimiter=',')
        all_counts.append(voxlet_counts)
    all_counts = np.vstack(all_counts)

    print "Now forming full array..."
    max_vox_id = all_counts[:, 0].max()
    accumulators[model_idx] = np.zeros(max_vox_id+1, int)
    for voxlet_idx, val in all_counts:
        accumulators[model_idx][voxlet_idx] += val
        
    accumulators[model_idx] = accumulators[model_idx].astype(float) *         (4500.0/accumulators[model_idx].sum())


# In[2]:

sorted_idxs = np.argsort(accumulators[0])[::-1]
plt.figure(figsize=(15, 15))
print np.sum(sorted_idxs[:4500] > 225000)

print np.sum(sorted_idxs[:4500] < 225000)


# In[3]:

# combining the two accumulators together...
full_counts = np.hstack((accumulators[0], accumulators[1]))
full_model_idxs = np.hstack((accumulators[0] * 0, accumulators[1] * 0 + 1)).astype(np.int32)
original_data_idxs = np.hstack((np.arange(accumulators[0].shape[0], dtype=np.int32), 
                                np.arange(accumulators[1].shape[0], dtype=np.int32)))


plt.plot(np.sort(full_counts)[::-1][:500])
plt.ylabel('Frequency')
plt.xlabel('Voxlets')
plt.ylim(ymin=0)

# Some axis formatting...
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines['left'].set_color([0.7, 0.7, 0.7])
ax.spines['bottom'].set_color([0.6, 0.6, 0.6])
ax.spines['left'].set_zorder(-10)


# In[4]:

import cPickle as pickle
with open('/media/ssd/data/oisin_house/models_full_split_floating_different_split/models/oma_cobweb.pkl', 'rb') as f:
    model_short = pickle.load(f)

print model_short.pca.components_.shape


# In[6]:

with open('/media/ssd/data/oisin_house/models_full_split_tall_different_split/models/oma_tall_cobweb.pkl', 'rb') as f:
    model_tall = pickle.load(f)
    
models = [model_short, model_tall]


# In[12]:

for m in models:
    m.forest = None
    print m.training_X.shape, m.training_X.dtype
    print m.training_Y.shape, m.training_Y.dtype
    print m.masks_pca.components_.shape, m.masks_pca.components_.dtype
    m.masks_pca = None


# In[13]:

print 450000*(80+400+80+400)*2 / 1e9
print 4 * 400* 40000 * 8 / 1e9


# In[14]:

# now rendering the top few...
idxs_to_render = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 1000, 150000] 
# idxs_to_render = [150000]
savepath = paths.data_folder + '/predictions/top_voxlets/%010d.png'

sorted_idxs = full_counts.argsort()[::-1]
        

from common import voxlets
from copy import copy


# for idx in sorted_idxs[:40]:
for rank in idxs_to_render:
    
    what_to_plot = copy(rank)
#     what_to_plot += (1 - abs(9-what_to_plot)) * 2

    idx = copy(sorted_idxs[what_to_plot])
    print rank, idx < 225000  
#     print full_counts[idx]
#     print full_model_idxs[idx]
#     print full_model_idxs[idx]
    model = models[full_model_idxs[idx]]
#     print original_data_idxs[idx]
    
    this_voxlet = model.training_Y[original_data_idxs[idx]]

    V = model.pca.inverse_transform(this_voxlet).reshape(model.voxlet_params['shape'])

    voxlets.render_single_voxlet(V, savepath % rank, 0, 1-full_model_idxs[idx])
    
    
    


# In[6]:

print full_counts.shape


# In[ ]:



