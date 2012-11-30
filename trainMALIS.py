import sys, time, h5py
import numpy as np
import cPickle as pickle
from h5utils import rH5data, wH5data
from scipy.io import loadmat, savemat
import tables
from pymalis import malisLoss
from tree_feature import TreeFeature
from sklearn.externals import joblib
from patch_datamatrix import extract_patches_3d, extract_patches_3d_affinity

n_iter = 2500
eta = 0.01
patch_size = (51,)*3
half_patch_size = tuple([i/2 for i in patch_size])
patch_stride_feature = (10,)*3
subimage_size = (25,)*3
subimage_extract_size = tuple([subimage_size[i]+patch_size[i]-1 for i in range(len(patch_size))])
n_output = 3

# forestFile = "results/unsup_sophie2_highSNR_n5_d8_final.pkl"
# forestFile = "results/unsup_sophie2_highSNR_n20_d10_final.pkl"
# forestFile = "results/unsup_sophie2_highSNR_n10_d10_gbrt_final.pkl"
# forestFile = "results/unsup_sophie2_highSNR_n50_d8_final.pkl"
# forestFile = "results/affgraph_e2198_t256_n20_d10_psz51_pstrd10_pstrdy5_gbrt_final.pkl"
forestFile = "results/affgraph_e2198_t256_n50_d20_psz51_pstrd10_pstrdy10_gbrt_final.pkl"
forest = joblib.load(forestFile)

dataset = "e2198_t256"
# dataset = "sophie2_highSNR"
inFile = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + ".mat"
inFile2 = "/nfs/home2/sturaga/net_sets/pymalis/" + dataset + "_features_small.h5"

# load the big image data cube and segmentation
t0 = time.time()
print "loading features and ground truth"
seg = loadmat(inFile)["seg"].astype(np.int32)
inFile2_h = h5py.File(inFile2,'r')
features = inFile2_h['volume/data']
print "  loaded features: %r, %r" % (features.dtype, features.shape)
print "  loaded seg:   %r, %r" % (seg.dtype, seg.shape)
print "done. elapsed time %0.3f s" % ((time.time()-t0))
maxsize = tuple([features.shape[i]-patch_size[i]+1-subimage_size[i]+1 for i in range(3)])

nhood = -np.identity(3, dtype=np.float64).copy(order='F')

tf = [TreeFeature(t, n_output=n_output, dtype="float32",C=0.001) for t in forest]
conn = np.zeros(subimage_size+(n_output,), dtype="float32", order="F")
dloss = np.zeros(subimage_size+(n_output,), dtype="float32", order="F")
seg_sub = np.zeros(subimage_size, dtype="uint16", order="F")
err = np.zeros(n_iter)

# rng = np.random.randint(sys.maxint)
for it in range(n_iter):

    # pick a sub-image
    sys.stdout.write("Extracting sub-image... ")
    t0 = time.time()
    idx = [np.random.randint(maxsize[i]) for i in range(3)]
    X = extract_patches_3d(image=features[idx[0]:idx[0]+subimage_extract_size[0],
                                          idx[1]:idx[1]+subimage_extract_size[1],
                                          idx[2]:idx[2]+subimage_extract_size[2],:],
                            patch_size=patch_size,
                            patch_stride=patch_stride_feature,
                            max_patches=1.0,
                            order = "F",
                            dtype = "float32")
    X.shape = (X.shape[0],-1)
    conn.fill(0.0)
    for t in tf:
        conn += t.predict(X).reshape(conn.shape)
    # conn.shape = subimage_size + (n_output,)
    seg_sub[:,:,:] = seg[ idx[0]+half_patch_size[0]:idx[0]+half_patch_size[0]+subimage_size[0],
                      idx[1]+half_patch_size[1]:idx[1]+half_patch_size[1]+subimage_size[1],
                      idx[2]+half_patch_size[2]:idx[2]+half_patch_size[2]+subimage_size[2] ]
    print "done. (elapsed time %0.3f s)" % ((time.time()-t0))

    # compute loss
    sys.stdout.write("Computing loss... ")
    t0 = time.time()
    pos = True
    neg = False
    dloss.fill(0.0)
    dloss, loss1, classErr1, randIndex1 = malisLoss(conn, nhood, seg_sub, dloss, 0.1, pos, neg)
    pos = False
    neg = True
    dloss, loss2, classErr2, randIndex2 = malisLoss(conn, nhood, seg_sub, dloss, 0.1, pos, neg)
    dloss /= 2
    print "done. (elapsed time %0.3f s)" % ((time.time()-t0))

    # update weights
    sys.stdout.write("Updating weights... ")
    t0 = time.time()
    dY = extract_patches_3d(dloss,
                            patch_size=(1,)*3,
                            patch_stride=(1,)*3,
                            max_patches=1.0,
                            order = "C",
                            dtype = "float32")
    dY.shape = (dY.shape[0],-1)
    for t in tf:
        t.gradient_update(X,dY,eta)
    print "done. (elapsed time %0.3f s)" % ((time.time()-t0))

    print "[iter %r] loss: %0.5f, classerr: %0.5f" % (it, loss1+loss2, classErr1+classErr2),
    print "(elapsed time %0.3f s)" % ((time.time()-t0))
