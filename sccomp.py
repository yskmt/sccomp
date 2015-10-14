import os
from pdb import set_trace as st

import numpy as np
import numpy.matlib as matlib
from skimage import data, io, transform
import matplotlib.pyplot as plt
import pickle

import lmgist
from lmgist import param_gist, show_gist
import mask_tk as mtk

from pb import poisson_blend, create_mask


gists_db_name = 'dbs/gist_data.npy'
f_db_name = 'dbs/file_names.npy'

data_dir = 'data/scene_database/'
img_dirs = [d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))]

mask_name = 'mask.png'

# param = param_gist()
# param.img_size = 256
# param.orientations_per_scale = [8, 8, 8, 8]
# param.number_blocks = 4
# param.fc_prefilt = 4


def create_gist_database(data_dir, img_dirs):
    """Create a GIST database"""

    if (not os.path.isfile(gists_db_name)) or (not os.path.isfile(f_db_name)):

        gist_data = []
        file_names = []
        for dir in img_dirs:
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

        with open('gistparams', 'w') as f:
            pickle.dump(param, f)
    else:
        print '%s and %s already exist' % (gists_db_name, f_db_name)

    return


def load_gists(gist_db_name, f_db_name):
    # load saved GIST features
    gist_data = np.load(gists_db_name)
    file_names = np.load(f_db_name)
    gist_data = gist_data.reshape((2688, 512)).T
    with open('gistparams', 'r') as f:
        param = pickle.load(f)

    return gist_data, file_names, param


def load_queries(img_dirs, data_dir, num_img, mask_name):
    """
    Load query image
    """

    img_files = []
    for dir in img_dirs:
        scene_dir = os.path.join(data_dir, dir)
        img_files += [os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
                      if os.path.isfile(os.path.join(scene_dir, f))
                      and 'jpg' in f]
    try:
        query_name = img_files[num_img]
    except IndexError:
        print "choose from 0 ~ %d!" % len(img_files)

    img_query = io.imread(query_name)

    return query_name, img_query, img_files


def get_gist(query_name, img_query, img_mask, param):
    '''
    Get the gist descriptor for the query image with mask
    '''

    # get the gist parameter of the source file
    gist, param = lmgist.lmgist(query_name, param)

    # resize the query and mask images to match the one used to compute
    # the GIST descriptor
    img_query = transform.resize(img_query, (param.img_size, param.img_size),
                                 clip=True, order=1)
    img_mask = transform.resize(img_mask, (param.img_size, param.img_size),
                                clip=True, order=1)

    # make sure the mask is 2d and in [0, 1]
    if len(img_mask.shape) != 2:
        raise TypeError('Mask image needs to be 2D!')
    elif not ((np.min(img_mask) == 0) and (np.max(img_mask == 1))):
        img_mask = (img_mask - np.min(img_mask)) \
            / (np.max(img_mask) - np.min(img_mask))

    # GIST weights are the average value of mask pixels within each block
    s = (np.linspace(0, param.img_size, param.number_blocks + 1)).astype(int)
    block_weight = np.zeros((param.number_blocks, param.number_blocks))
    for y in range(param.number_blocks):
        for x in range(param.number_blocks):
            block = img_mask[s[y]:s[y + 1], s[x]:s[x + 1]]
            block_weight[y, x] = np.mean(block)

    block_weight = 1 - block_weight
    n_filters = sum(param.orientations_per_scale)
    block_weight = matlib.repmat(
        np.reshape(block_weight.T, [param.number_blocks ** 2, 1]), n_filters, 1)

    return gist, block_weight


def get_matches(gist_data, file_names, img_mask):
    """
    Using the gist value, get the closes matches from the database
    """

    # find the close matches of a scene chosen using the
    er = np.linalg.norm(
        (gist_data - gist.reshape((len(gist_data), 1))) * block_weight, axis=0).flatten()
    agst = np.argsort(er)

    file_names = np.array(file_names)
    fn_sorted = file_names[agst][1:6]
    print '\nImages chosen from scene completion are:\n', fn_sorted

    img_target = io.imread(query_name).astype(np.float64)
    img_sc = poisson_blend(img_mask, img_mask, img_target,
                           method='src', offset_adj=(0, 0))
    print '\nOriginal image with mask.'
    plt.imshow(img_sc)
    plt.show()

    print '\nCompleted images from 1st to 5th choices:'
    for i in range(len(fn_sorted)):
        # img_target = io.imread(query_name).astype(np.float64)
        img_src = io.imread(fn_sorted[i]).astype(np.float64)
        img_mask = io.imread(mask_name, as_grey=True)
        offset = (0, 0)

        img_mask, img_src, offset_adj \
            = create_mask(img_mask.astype(np.float64),
                          img_target, img_src, offset=offset)

        img_sc = poisson_blend(img_mask, img_src, img_target, method='normal',
                               offset_adj=offset)
        plt.imshow(img_sc)
        plt.show()

    return fn_sorted


create_gist_database(data_dir, img_dirs)

gist_data, file_names, param = load_gists(gists_db_name, f_db_name)

# ask for the image number to work with
print "Choose the image number"
num_img = int(raw_input())

query_name, img_query, img_files \
    = load_queries(img_dirs, data_dir, num_img, mask_name)

uin = 'Y'
if os.path.exists(mask_name):
    print "Mask file already exist. Create a new one? (Y|n)"
    uin = raw_input()
if uin in ['Y', '']:
    mtk.create_mask(query_name, mask_name)

img_mask = io.imread(mask_name, as_grey=True)

gist, block_weight = get_gist(query_name, img_query, img_mask, param)

matches = get_matches(gist_data, file_names, img_mask)

# show_gist(gd, param)
