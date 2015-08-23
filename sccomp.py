import os

import numpy as np
from skimage import data, io, transform
import matplotlib.pyplot as plt

# import gist
import lmgist

gists_db_name = 'dbs/gist_data.npy'
f_db_name = 'dbs/file_names.npy'

# create a GIST database
imsize = (128, 128)
img_dir = 'lear-gist-python/sample/spatial_envelope_256x256_static_8outdoorcategories'
fnames = []
gists = []

# num_f = len(os.listdir(img_dir))
# i = 0
# for f in os.listdir(img_dir):
#     if (i % 100) == 0:
#         print i, num_f
#     i += 1

#     fnames.append(f)

#     try:
#         img = io.imread(img_dir + '/' + f)
#     except:
#         print f

#     img_resized = transform.resize(
#         img, imsize, preserve_range=True).astype(np.uint8)
#     gists.append(gist.extract(img_resized))

# gists = np.array(gists)
# np.save(gists_db_name, gists)
# np.save(f_db_name, fnames)

# load saved GIST features
gists = np.load(gists_db_name)
fnames = np.load(f_db_name)

# load a image and find the best 5 images from database
scene_db = 'data/scene_database'
img = io.imread(scene_db + '/coast/nat202.jpg')
# img = io.imread('data/test_set/IMG_4630.bmp')
img_resized = transform.resize(
    img, imsize, preserve_range=True).astype(np.uint8)
fig = plt.figure(1)
ax = fig.add_subplot(111)


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print 'click', event
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

implt = ax.imshow(img_resized)
linebuilder = LineBuilder(implt)
plt.show()



# desc = gist.extract(img_resized)

# er = []
# for g in gists:
#     er.append(np.linalg.norm(g - desc))
# idx_sorted = np.argsort(er)

# plt.figure(2)
# for i in range(5):
#     print fnames[idx_sorted[i]], er[idx_sorted[i]]
#     plt.imshow(io.imread(img_dir + '/' + fnames[idx_sorted[i]]))
#     plt.show()


