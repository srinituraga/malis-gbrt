# from __future__ import print_function
import sys, time
# sys.path.insert(1,'/nfs/home2/sturaga/ilastik/rfmalis/scikit-learn/')
import numpy as np
import sklearn, vigra, glob, pylab, h5py
import cPickle as pickle
from h5utils import rH5data, wH5data
from scipy.io import loadmat, savemat
from pymalis import malisLoss
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
# from gradient_boosted_rf2 import GradientBoostedRandomForest
from patch_datamatrix import extract_patches_3d

sample_density = 0.001

# inFile = "e2198_t256"
# inFile2 = "/nfs/home2/sturaga/data/E2198/" + inFile + ".mat"
# dataset = "sophie1_lowSNR"
dataset = "sophie2_highSNR"
inFile = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + ".mat"
inFile2 = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + "_features_small.h5"

# load the big image data cube and segmentation
t0 = time.time()
print "loading image, features, ground truth segmentation"
im = loadmat(inFile)["im"].astype(np.float32)
seg = loadmat(inFile)["seg"].astype(np.uint16)
print "  loaded im:   %r, %r" % (im.dtype, im.shape)
print "  loaded seg:   %r, %r" % (seg.dtype, seg.shape)
assert im.shape == seg.shape
print "  im has range [%f, %f]" % (np.min(im[:]), np.max(im[:]))
features = rH5data(inFile2,"volume/data")
print "  loaded features: %r, %r" % (features.dtype, features.shape)
print "done. elapsed time %0.3f s" % ((time.time()-t0))

t0 = time.time()
print "loading forest...",
forest = pickle.load(open("results/unsup_sophie2_highSNR_n10_d8_final.pkl", "rb"))
print "done. elapsed time %0.3f s" % ((time.time()-t0))

print "  Extracting patches...",
# rng = int(time.time()) #np.random.mtrand._rand
rng = np.random.randint(sys.maxint)
X = extract_patches_3d(image=features,
                        patch_size=(21,21,21),
                        patch_stride=(5,5,5),
                        max_patches=sample_density,
                        random_state = rng)
Y = extract_patches_3d(image=im,
                        patch_size=(21,21,21),
                        patch_stride=(1,1,1),
                        max_patches=sample_density,
                        random_state = rng)
X.shape = (X.shape[0],-1)
Y.shape = (Y.shape[0],-1)
print "done. elapsed time %0.3f s" % ((time.time()-t0))
