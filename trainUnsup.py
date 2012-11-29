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

n_estimators    = 10
max_depth       = 20
max_features    = "sqrt"
sample_density  = 100000 #0.0005
min_samples_leaf  = 100 
gradient_boosting = True

# inFile = "e2198_t256"
# inFile2 = "/nfs/home2/sturaga/data/E2198/" + inFile + ".mat"
# dataset = "sophie1_lowSNR"
dataset = "sophie2_highSNR"
inFile = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + ".mat"
inFile2 = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + "_features_small.h5"
outFile = "results/unsup_" + dataset + "_n" + `n_estimators` + "_d" + `max_depth`

# load the big image data cube and segmentation
t0 = time.time()
print "loading image, features, ground truth segmentation"
im = loadmat(inFile)["im"].astype(np.float32)
seg = loadmat(inFile)["seg"].astype(np.uint16)
print "  loaded im:   %r, %r" % (im.dtype, im.shape)
print "  loaded seg:   %r, %r" % (seg.dtype, seg.shape)
assert im.shape == seg.shape
print "  im has range [%f, %f]" % (np.min(im[:]), np.max(im[:]))
# features = rH5data(inFile2,"volume/data")
inFile2_h = h5py.File(inFile2,'r')
features = inFile2_h['volume/data']
print "  loaded features: %r, %r" % (features.dtype, features.shape)
print "done. elapsed time %0.3f s" % ((time.time()-t0))


forest = [ExtraTreeRegressor(max_depth=max_depth, min_density=0.0, max_features=max_features, min_samples_leaf=min_samples_leaf)
						for each in range(n_estimators)]
mse = ()
for k in range(n_estimators):

	# convert to datamatrix
	print "[tree #%r]" % k
	t0 = time.time()
	print "  Extracting patches...",
	# rng = np.random.mtrand._rand
	rng = np.random.randint(sys.maxint)
	X = extract_patches_3d(image=features,
							patch_size=(46,46,46),
							patch_stride=(9,9,9),
							max_patches=sample_density,
							order = "F",
							dtype = "float32",
							random_state = rng)
	Y = extract_patches_3d(image=im,
							patch_size=(46,46,46),
							patch_stride=(5,5,5),
							max_patches=sample_density,
							order = "C",
							dtype = "float64",
							random_state = rng)
	X.shape = (X.shape[0],-1)
	Y.shape = (Y.shape[0],-1)
	print "done. elapsed time %0.3f s" % ((time.time()-t0))

	# fit a tree
	t0 = time.time()
	print "  Fitting tree...",
	if gradient_boosting and k>0:
		Yp = np.zeros(Y.shape)
		for kk in range(k):
			Yp += forest[kk].predict(X)
		mse += (np.sqrt(np.mean(np.square(Yp-Y)))/2,)
		print "mse: ", mse[-1]
		forest[k].fit(X,0.95*(Y-Yp))
	else:
		forest[k].fit(X, Y)
	print "done. elapsed time %0.3f s" % ((time.time()-t0))

if gradient_boosting:
	pickle.dump(forest,open(outFile+"_gbrt_final.pkl","wb"),pickle.HIGHEST_PROTOCOL)
else:
	pickle.dump(forest,open(outFile+"_final.pkl","wb"),pickle.HIGHEST_PROTOCOL)

