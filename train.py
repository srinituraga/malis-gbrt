import sys
sys.path.insert(1,'/nfs/home2/sturaga/ilastik/rfmalis/scikit-learn/')
import sklearn
import numpy, vigra
import h5py
from h5utils import rH5data, wH5data
from pymalis import malisLoss
from sklearn.ensemble.forest import RandomForestRegressor 
import glob
import pylab
from gradient_boosted_rf2 import GradientBoostedRandomForest
from sklearn.ensemble import ExtraTreesRegressor
from scipy.io import loadmat, savemat
import cPickle as pickle


margin = 0.3
n_jobs = -1 #If -1, then the number of jobs is set to the number of cores.
min_density = 0.1
max_depth_initial = 8
n_estimators = 16
max_depth_gb = 8
n_estimators_gb = 2
verbose = 0
fancy_update_every_iter = 50
fancy_update_until_iter = 2e3
fancy_update_method = "add_forest"
cubeLength = 50
sub_sample_size = 1e5
sub_sample_iter = 0
eta = 1e3
alpha = 1e3
max_iterations = 5000
# inFile = "e2198_t256"
# inFile2 = "/nfs/home2/sturaga/data/E2198/" + inFile + ".mat"
# dataset = "sophie1_lowSNR"
dataset = "sophie2_highSNR"
inFile = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + ".mat"
inFile2 = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + "_features.h5"
outFile = "results/oneRF_" + dataset + "_n" + `n_estimators` + "_d" + `max_depth_initial`


# load the big data cube and segmentation
im = loadmat(inFile)["im"].astype(numpy.float32)
seg = loadmat(inFile)["seg"].astype(numpy.uint16)
assert im.shape == seg.shape
print "loaded data, ground truth segmentation with shape = %r" % (seg.shape,)
print "data has range [%f, %f]" % (numpy.min(im[:]), numpy.max(im[:]))

# conn = numpy.zeros(im.shape + (3,),dtype='float32', order='F')
# print "conn has shape = %r" % (conn.shape,)
nhood = -numpy.identity(3, dtype=numpy.float64).copy(order='F')

# print "conn:  %r, %r" % (conn.dtype, conn.shape)
print "nhood: %r, %r" % (nhood.dtype, nhood.shape)
print "seg:   %r, %r" % (seg.dtype, seg.shape)

# load features
features = rH5data(inFile2,"volume/data")

mask = numpy.zeros(seg.shape + (3,), numpy.uint8)
mask[1:, : , : , 0 ] = numpy.logical_or((seg[1:, : , :]  != 0), (seg[:-1, : , :]  != 0))
mask[: , 1:, : , 1 ] = numpy.logical_or((seg[:, 1: , :]  != 0), (seg[:, :-1 , :]  != 0))
mask[: , : , 1:, 2 ] = numpy.logical_or((seg[:, : , 1:]  != 0), (seg[:, : , :-1]  != 0))

labels = numpy.zeros(seg.shape + (3,), numpy.uint8)
labels[1:, : , : ,0 ] = (seg[:-1, :,   :  ] == seg[1:, :  , : ] )*mask[1:, : , : , 0]
labels[: , 1:, : ,1 ] = (seg[:,   :-1, :  ] == seg[:   ,1:, : ] )*mask[:, 1: , : , 1]
labels[: , : , 1:,2 ] = (seg[:,   :,   :-1] == seg[:   ,: , 1:] )*mask[:, : , 1: , 2]
# a hack to get a boundary map label
labels = numpy.all(labels, axis=3)


print "reshaping features to %r" % ((numpy.prod(features.shape[:-1]), features.shape[-1]),)
featuresShape = features.shape
features = numpy.reshape(features, (numpy.prod(featuresShape[:-1]), featuresShape[-1]))
labels = labels.ravel()
# mask_flat = mask.ravel()

# randomly sample the data points
idx = numpy.random.randint(0,features.shape[0],sub_sample_size)
features_slice = features[idx,:]
labels_slice   = labels[idx]
print "Learning initial pixel-wise random forests..."
clf = GradientBoostedRandomForest(base_regressor=ExtraTreesRegressor, oob_score = True, n_estimators = n_estimators, max_depth_gb=max_depth_gb, max_depth=max_depth_initial, n_jobs=n_jobs, verbose=verbose, n_estimators_gb = n_estimators_gb, min_density=min_density)
clf.fit(features_slice, labels_slice)
print "*** initial pickling parameters: %s ***" % outFile
pickle.dump(clf,open(outFile+"_init.pkl","wb"),pickle.HIGHEST_PROTOCOL)
print "  oob = %f" % clf.regressors[-1].oob_score_

stats = {'loss': numpy.zeros(max_iterations), \
          'classErr': numpy.zeros(max_iterations), \
          'randIndex': numpy.zeros(max_iterations)}

# plotting stuff
fg = pylab.figure()
pylab.ion()

features.shape = featuresShape
conn = numpy.zeros(seg.shape + (3,), dtype=numpy.float32)
dloss = numpy.zeros(seg.shape, dtype=numpy.float32)
conn_slice = numpy.zeros((cubeLength,cubeLength,cubeLength,3), dtype=numpy.float32, order='F')

for i in range(max_iterations):

    # subsample the dataset
    sl = []
    for k in range(3):
      offset = numpy.random.randint(0,featuresShape[k]-cubeLength)
      sl.append(slice(offset,offset+cubeLength))
    sl = tuple(sl)
    features_slice = features[sl].reshape(cubeLength**3,featuresShape[-1])
    seg_slice = seg[sl].copy(order='F').astype(numpy.uint16)
    pred = clf.predict(features_slice)
    pred.shape = seg_slice.shape
    for k in range(3):
      conn_slice[:,:,:,k] = pred
    conn_slice.flags

    pos = True
    neg = False
    dloss_pos = numpy.zeros(conn_slice.shape, dtype=numpy.float32, order="F")
    dloss_pos, loss1, classErr1, randIndex1 = malisLoss(conn_slice, nhood, seg_slice, dloss_pos, margin, pos, neg)
    pos = False
    neg = True
    dloss_neg = numpy.zeros(conn_slice.shape, dtype=numpy.float32, order="F")
    dloss_neg, loss2, classErr2, randIndex2 = malisLoss(conn_slice, nhood, seg_slice, dloss_neg, margin, pos, neg)
    dloss_slice = (dloss_pos + dloss_neg) / 2


    loss = (loss1 + loss2) / 2
    classErr = (classErr1 + classErr2) / 2
    randIndex = (randIndex1 + randIndex2)/2
    stats['loss'][i] = loss
    stats['classErr'][i] = classErr
    stats['randIndex'][i] = randIndex

    print "MAX Gradient = %f, MIN Gradient = %f" % (numpy.max(dloss_slice), numpy.min(dloss_slice))

    conn[sl] = conn_slice
    dloss[sl] = numpy.sum(dloss_slice,axis=3)

    gradient_slice = (eta/(alpha+i))*numpy.sum(dloss_slice,axis=3).ravel()

    if (i < fancy_update_until_iter) and (i % fancy_update_every_iter) == (fancy_update_every_iter-1):
      update_method = fancy_update_method
    else:
      update_method = "leaves"

    print "USING UPDATE METHOD : %s " % update_method
    if update_method == "leaves":
      clf.update_leaves(features_slice,gradient_slice)
    #   # clf.update_leaves(gradient_flat[numpy.where(mask_flat)[0]])
    # elif update_method == "replace_forest":  
    #   clf.update_replace_forest(gradient_flat[numpy.where(mask_flat)[0]])
    # elif update_method == "add_forest_percentile":  
    #   clf.update_add_forest_percentile(gradient_flat[numpy.where(mask_flat)[0]])
    # elif update_method == "add_forest_nz":
    #   clf.update_add_forest_nz(gradient_flat[numpy.where(mask_flat)[0]])
    # elif update_method == "add_forest_bins":
    #   clf.update_add_forest_bins(gradient_flat[numpy.where(mask_flat)[0]])
    elif update_method == "add_forest":
      # p = 0.001 + numpy.abs(dloss.ravel()); p = p/numpy.sum(p)
      # # idx = numpy.random.choice(p.shape[0],size=sub_sample_size,p=p)
      # idx = p.cumsum().searchsorted(numpy.random.sample(sub_sample_size))
      # features.shape = (numpy.prod(featuresShape[:-1]),featuresShape[-1])
      # dloss.shape = (features.shape[0],)
      # features_slice = features[idx,:]
      # gradient_slice = (eta/(alpha+i))*dloss_slice[idx]
      clf.update_add_forest(features_slice,gradient_slice)
      # features.shape = featuresShape
      # dloss.shape = featuresShape[:-1]
    else:
      raise RuntimeError("Unknown update method %r" % update_method)
    
    print "iteration = %d, loss = %f, classErr = %f, randIndex = %f, minDloss = %f, maxDloss = %f" % (i, loss, classErr, randIndex, numpy.min(dloss_slice), numpy.max(dloss_slice))

    if i % 50 == 0:
        print "*** writing conn and pickling parameters: %s ***" % outFile
        savemat(outFile+".mat",{"conn":conn,"loss":stats["loss"][:i],"classErr":stats["classErr"][:i]})
        pickle.dump(clf,open(outFile+"_final.pkl","wb"),pickle.HIGHEST_PROTOCOL)

    if i % 5 == 0:
        gc = pylab.gca()
        pylab.subplot(121)
        pylab.hold(False)
        pylab.plot(stats['loss'][:i])
        pylab.xlabel("iteration")
        pylab.ylabel("loss")
        pylab.subplot(122)
        pylab.hold(False)
        pylab.plot(stats['classErr'][:i])
        pylab.xlabel("iteration")
        pylab.ylabel("classification error")
        pylab.draw()
