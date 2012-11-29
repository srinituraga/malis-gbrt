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
from patch_datamatrix import extract_patches_3d, extract_patches_3d_affinity

n_estimators    = 50
max_depth       = 20
max_features    = "sqrt"
sample_density  = 200000 #0.0005
min_samples_leaf  = 200
gradient_boosting = True
patch_size 		= (51,51,51)
patch_stride 	= (10,10,10)
patch_stride_y 	= (5,5,5)
longrange_affinity = True

dataset = "e2198_t256"
# dataset = "sophie1_lowSNR"
# dataset = "sophie2_highSNR"
inFile = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + ".mat"
inFile2 = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + "_features_small.h5"
outFile = "results/affgraph_" + dataset \
				+ "_n" + `n_estimators` \
				+ "_d" + `max_depth` \
				+ "_psz" + `patch_size[0]` \
				+ "_pstrd" + `patch_stride[0]` \
				+ "_pstrdy" + `patch_stride_y[0]`
print "dataset: " + dataset
print "outFile: " + outFile
print "sample_density: %r" % sample_density 
print "patch_size: " + `patch_size`
print "patch_stride: " + `patch_stride`
print "patch_stride_y: " + `patch_stride_y`

# load the big image data cube and segmentation
t0 = time.time()
print "loading image, features, ground truth segmentation"
if longrange_affinity:
	label = loadmat(inFile)["seg"].astype(np.int32)
	extractfun = extract_patches_3d_affinity
else:
	label = loadmat(inFile)["conn"].astype(np.float32)
	extractfun = extract_patches_3d
print "  loaded label:   %r, %r" % (label.dtype, label.shape)
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
							patch_size=patch_size,
							patch_stride=patch_stride,
							max_patches=sample_density,
							order = "F",
							dtype = "float32",
							random_state = rng)
	Y = extractfun(label,
							patch_size=patch_size,
							patch_stride=patch_stride_y,
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

