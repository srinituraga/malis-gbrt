from patch_datamatrix import extract_patches_3d, extract_patches_3d_affinity
import sys, time, h5py
import cPickle as pickle
from h5utils import rH5data, wH5data
from scipy.io import loadmat, savemat
import tables
from pymalis import malisLoss
from tree_feature import TreeFeature

n_iter = 2500
# sample_density = 0.00001
n_patches = 100
eta = 0.1/n_patches
patch_size = (51,51,51)
patch_stride_feature = (10,10,10)
patch_idx_keep = (tuple(range(0,51,10)),)*3
# patch_idx_keep = (tuple(range(21,30)),)*3
# patch_idx_keep = (tuple(range(23,28)),)*3
# patch_idx_keep = (tuple(range(19,32,3)),)*3
# patch_idx_keep = (tuple( [25+i for i in [-5, -2, 0, 2, 5]] ),)*3
n_output = np.prod([len(l) for l in patch_idx_keep])
longrange_affinity = True

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
if longrange_affinity:
    label = loadmat(inFile)["seg"].astype(np.int32)
    extractfun = extract_patches_3d_affinity
else:
    label = loadmat(inFile)["conn"].astype(np.float32)
    extractfun = extract_patches_3d
inFile2_h = h5py.File(inFile2,'r')
features = inFile2_h['volume/data']
print "  loaded features: %r, %r" % (features.dtype, features.shape)
print "  loaded label:   %r, %r" % (label.dtype, label.shape)
print "done. elapsed time %0.3f s" % ((time.time()-t0))


tf = [TreeFeature(t,n_output=n_output,dtype="float32",C=0.001) for t in forest]
Yp = np.array([])
err = np.zeros(n_iter)

# rng = np.random.randint(sys.maxint)
for it in range(n_iter):
    t0 = time.time()
    rng = np.random.randint(sys.maxint)
    X = extract_patches_3d(image=features,
                            patch_size=patch_size,
                            patch_stride=patch_stride_feature,
                            max_patches=n_patches,
                            order = "F",
                            dtype = "float32",
                            random_state = rng)
    Y = extractfun(label,
                            patch_size=patch_size,
                            patch_idx_keep=patch_idx_keep,
                            max_patches=n_patches,
                            order = "C",
                            dtype = "float32",
                            random_state = rng)
    # Y = Y[:,13]
    X.shape = (X.shape[0],-1)
    Y.shape = (Y.shape[0],-1)

    Yp.resize(Y.shape)
    Yp.fill(0)
    for t in tf:
        Yp += t.predict(X)
    err[it] = tf[0].mse(Y,Yp)
    dY = Yp-Y
    for t in tf:
        t.gradient_update(X,dY,eta)

    print "[iter %r] MSE: %0.5f" % (it, err[it]),
        print "(elapsed time %0.3f s)" % ((time.time()-t0))
