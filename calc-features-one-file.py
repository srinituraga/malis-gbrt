import vigra
import math
import numpy
import mahotas
import os
import h5py
import sys
import gc
from h5utils import rH5data, wH5data
from scipy.io import loadmat, savemat

#slicing = (slice(0,100,None), slice(0,101), slice(0,102))
# cubeLength = 255
# offset = 0
# slicing = (slice(offset,offset+cubeLength), slice(offset,offset+cubeLength), slice(offset,offset+cubeLength))

# inFile = "e1088_roi8"
inFile = "e2198_t256"
# inFile = "sophie1_lowSNR"
# inFile = "sophie2_highSNR"
outFile = "/nfs/home2/sturaga/net_sets/pymalis/" + inFile + "_features_small.h5"
# d = loadmat("/nfs/home2/sturaga/net_sets/pymalis/" + inFile + ".mat")["im"][slicing].astype(numpy.float32)
d = loadmat("/nfs/home2/sturaga/net_sets/pymalis/" + inFile + ".mat")["im"].astype(numpy.float32)


def gradient(d,s):
    return vigra.filters.gaussianGradientMagnitude(d, s)[...,numpy.newaxis]
def hessianEV(d,s):    
    return vigra.filters.hessianOfGaussianEigenvalues(d,s)#[:,:,:,0:1]
def hessian(d,s):    
    return vigra.filters.hessianOfGaussian3D(d,s)#[:,:,:,0:1]
def gaussianSmoothing(d,s):
    return vigra.filters.gaussianSmoothing(d,s)[...,numpy.newaxis]
def gradient(d,s):
    return vigra.filters.gaussianGradient(d,s)
def laplacianOfGaussian(d,s):
    return vigra.filters.laplacianOfGaussian(d,s)[...,numpy.newaxis]

def haralick_roti(d,size):
    indices_flat = numpy.arange(d.size)
    shape = numpy.asarray(d.shape)
    result = numpy.ndarray((d.size,13),numpy.float32)

    for ind in indices_flat.flatten():
      if ind % 500 == 0:
          sys.stdout.write("\r    Haralick : %f%%   " % (100.0*ind / d.size))
          sys.stdout.flush()
      pos = numpy.asarray(numpy.unravel_index(ind, shape))
      start = pos - size
      start = numpy.maximum(start, 0)
      stop = pos + size
      stop = numpy.minimum(stop, shape)
      res = mahotas.features.haralick(d[start[0]:stop[0],start[1]:stop[1],start[2]:stop[2]].astype(numpy.uint8))
      result[ind,:] =  numpy.average(res, axis = 0)
    sys.stdout.write("\n")

    result = numpy.reshape(result, d.shape + (13,))
    return result

# scales = [0.5, 0.9, 1.2, 1.5, 1.9, 2.0, 2.5, 3.0, 3.5]
# scales = [0.5, 0.9, 1.2, 1.5, 1.9, 2.5, 3.5]
scales = [1, 3]
# haralick_scales = [1,3,5]
featureFunctors = [gaussianSmoothing, gradient, hessian]
nfeat =       sum([                1,        3,       6])
nfeat *= len(scales)
# featureFunctors = [(scales, gaussianSmoothing), (scales, gradient), (scales, hessianEV), (scales, laplacianOfGaussian), (scales, hessian)]
# featureFunctors = [(haralick_scales, haralick_roti),(scales, hessianEV0), (scales, gaussianSmoothing)]#, gradient, laplacianOfGaussian]
# featureFunctors = [(scales, gradient)]

features = numpy.ndarray(d.shape+(nfeat,),numpy.float32)
print "Creating features, features shape :" , features.shape
nfeatEnd = 0
nfeatStart = 0
for s in scales:
  for f in featureFunctors:
    # fname = ""
    # if type(s) == float:
    #     fname = "%s_%s_%.03f.h5" % (outFile, f.__name__, s)
    # else:
    #     fname = "%s_%s_%d.h5" % (outFile, f.__name__, s)

    print "Calculating feature: " , f.__name__ , ", scale: " , str(s) 
    w = f(d, s)
    if w is None:
      continue
    assert w.ndim == 4
    nfeatEnd = nfeatEnd+w.shape[3]
    print " copying features... " , "nfeatStart: " , str(nfeatStart) , ", nfeatEnd: " , str(nfeatEnd)
    print "feature slice shape:" , features[:,:,:,nfeatStart:nfeatEnd].shape , ", w shape:" , w.shape
    features[:,:,:,nfeatStart:nfeatEnd] = w
    nfeatStart = nfeatEnd
    gc.collect()
    # if features == []:
    #   features = [w]
    # else:
    #   print " concatenating features... " , "#feat: " , str(features.shape[3]) , ", #newfeat: " , str(w.shape[3])
    #   features = numpy.concatenate((features,w),axis=3)
#     print " appending features... " , "#feat: " , str(nfeat)
#     features.append(w)
# print "\nConcatenating features... " , "#feat: " , str(nfeat)
# features = numpy.concatenate(features,axis=3)
wH5data(outFile, "volume/data", features[:,:,:,:], verbose=True, overwrite=True)

