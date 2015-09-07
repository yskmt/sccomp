import os
from pdb import set_trace as st

import numpy as np
import numpy.matlib as matlib
from skimage import data, io, transform
import matplotlib.pyplot as plt

import lmgist
from lmgist import param_gist, show_gist
import mask_tk as mtk

import sys
sys.path.append('../pb')
from pb import poisson_blend, create_mask

gists_db_name = 'dbs/gist_data.npy'
f_db_name = 'dbs/file_names.npy'

# create a GIST database
param = param_gist()
param.img_size = 256
param.orientations_per_scale = [8, 8, 8, 8]
param.number_blocks = 4
param.fc_prefilt = 4

if (not os.path.isfile(gists_db_name)) or (not os.path.isfile(f_db_name)):

    data_dir = '/Users/ysakamoto/Projects/sccomp/data/scene_database/'
    dirs = [d for d in
            os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))]

    gist_data = []
    file_names = []
    for dir in dirs:
        scene_dir = os.path.join(data_dir, dir)
        img_files = [f for f in os.listdir(scene_dir)
                     if os.path.isfile(os.path.join(scene_dir, f))
                     and 'jpg' in f]

        for f in img_files:
            img_file = os.path.join(scene_dir, f)

            print img_file
            gist, param \
                = lmgist.lmgist(img_file, param)

            gist_data.append(gist)
            file_names.append(img_file)

    np.save(gists_db_name, gist_data)
    np.save(f_db_name, file_names)

else:
    print '%s and %s already exist' % (gists_db_name, f_db_name)

# load saved GIST features
gist_data = np.load(gists_db_name)
file_names = np.load(f_db_name)
 
# load query image and mask
# query_name = 'data/scene_database/coast/n238045.jpg'
# query_name = 'data/scene_database/inside_city/boston247.jpg'
query_name = 'data/scene_database/street/urb848.jpg'
mask_name = 'mask.png'
img_query = io.imread(query_name)
if not os.path.isfile(mask_name):
    mtk.create_mask(query_name, mask_name)
img_mask = io.imread(mask_name, as_grey=True)

gist, param = lmgist.lmgist(query_name, param)

# resize the query and mask images to match the one used to compute
# the GIST descriptor
img_query = transform.resize(img_query, (param.img_size, param.img_size),
                             clip=True, order=1)
img_mask = transform.resize(img_mask, (param.img_size, param.img_size),
                            clip=True, order=1)

# make sure the mask is 2d and in {0, 1}
if len(img_mask.shape) != 2:
    raise TypeError('Mask image needs to be 2D!')
elif not ((np.min(img_mask) == 0) and (np.max(img_mask == 1))):
    img_mask = (img_mask - np.min(img_mask)) \
        / (np.max(img_mask) - np.min(img_mask))
    
# GIST weights are the average value of mask pixels within each block
s = (np.linspace(0, param.img_size, param.number_blocks+1)).astype(int)
block_weight = np.zeros((param.number_blocks, param.number_blocks))
for y in range(param.number_blocks):
    for x in range(param.number_blocks):
        block = img_mask[s[y]:s[y+1], s[x]:s[x+1]]
        block_weight[y, x] = np.mean(block)

block_weight = 1 - block_weight
n_filters = sum(param.orientations_per_scale)
block_weight = matlib.repmat(
    np.reshape(block_weight.T, [param.number_blocks**2,1]), n_filters, 1)

# analysis: find the close matches of a scene chosen using the
# weighted Euclidean distance
# k = 1001
# gd = gist_data[k]
# fn = file_names[k]

gist_data = gist_data.reshape((2688, 512)).T

er = np.linalg.norm(
    (gist_data - gist.reshape((len(gist_data), 1)))*block_weight, axis=0).flatten()
agst = np.argsort(er)

file_names = np.array(file_names)
fn_sorted = file_names[agst][:10]
print fn_sorted

# img_st = io.imread(query_name)
# plt.imshow(img_st)
# plt.show()

for i in range(len(fn_sorted)):
    # img_st = io.imread(fn_sorted[i])
    # plt.imshow(img_st)
    # plt.show()

    # img_sc = img_query
    # img_sc[np.where(img_mask==1)] = img_st[np.where(img_mask==1)]

    img_target = io.imread(query_name).astype(np.float64)
    img_src = io.imread(fn_sorted[i]).astype(np.float64)
    img_mask = io.imread(mask_name, as_grey=True)
    offset = (0,0)
    
    img_mask, img_src, offset_adj \
        = create_mask(img_mask.astype(np.float64),
                      img_target, img_src, offset=offset)

    img_sc = poisson_blend(img_mask, img_src, img_target, method='normal', offset_adj=offset)
    plt.imshow(img_sc)
    plt.show()

    
    
# show_gist(gd, param)

