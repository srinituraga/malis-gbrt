import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.tree import ExtraTreeRegressor

_TREE_LEAF = -1

class TreeFeature:

    def __init__(self, tree, n_output, dtype, C=0.0, alpha=0.1):
        self.n_output = n_output
        self.feature = tree.tree_.feature
        self.threshold = tree.tree_.threshold
        self.children_left = tree.tree_.children_left
        self.children_right = tree.tree_.children_right
        self.value = np.zeros((tree.tree_.value.shape[0],n_output),dtype=dtype)
        self.grad_accum = np.ones((tree.tree_.value.shape[0],n_output),dtype=dtype)
        self.last_visit = np.zeros((tree.tree_.value.shape[0],n_output),dtype="int32")
        self.counter = 0
        self.C = C
        self.alpha = alpha

    def predict(self,X):
        """Predict target for X."""
        n_samples = X.shape[0]
        out = np.zeros((n_samples,self.n_output),dtype=self.value.dtype)

        for i in range(n_samples):
            node_id = 0
            for k in range(self.n_output):
                out[i,k] = self.value[node_id,k]

            # While node_id not a leaf
            while self.children_left[node_id] != _TREE_LEAF: # and self.children_right[node_id] != _TREE_LEAF:
                if X[i, self.feature[node_id]] <= self.threshold[node_id]:
                    node_id = self.children_left[node_id]
                else:
                    node_id = self.children_right[node_id]
                for k in range(self.n_output):
                    out[i,k] += self.value[node_id,k]

        return out

    def score(self,X):
        return np.mean(np.square(Y-predict(X))/2)

    def mse(self,Y,Yp):
        return np.mean(np.square(Y-Yp)/2)

    def gradient_update(self,X,dY,eta):
        n_samples = X.shape[0]

        for i in range(n_samples):
            self.counter += 1
            node_id = 0
            depth = 0
            for k in range(self.n_output):
                grad = dY[i,k] + self.C*np.exp(self.alpha*depth)*np.sign(self.value[node_id,k])*(self.counter-self.last_visit[node_id,k])
                self.grad_accum[node_id,k] += grad*grad
                self.value[node_id,k] -= eta*grad/np.sqrt(self.grad_accum[node_id,k])
                self.last_visit[node_id,k] = self.counter

            # While node_id not a leaf
            while self.children_left[node_id] != _TREE_LEAF: # and self.children_right[node_id] != _TREE_LEAF:
                depth += 1
                if X[i, self.feature[node_id]] <= self.threshold[node_id]:
                    node_id = self.children_left[node_id]
                else:
                    node_id = self.children_right[node_id]
                for k in range(self.n_output):
                    grad = dY[i,k] + self.C*np.exp(self.alpha*depth)*np.sign(self.value[node_id,k])*(self.counter-self.last_visit[node_id,k])
                    self.grad_accum[node_id,k] += grad*grad
                    self.value[node_id,k] -= eta*grad/np.sqrt(self.grad_accum[node_id,k])
                    self.last_visit[node_id,k] = self.counter


if __name__ == "__main__":
    from patch_datamatrix import extract_patches_3d, extract_patches_3d_affinity
    import sys, time, h5py
    import cPickle as pickle
    from h5utils import rH5data, wH5data
    from scipy.io import loadmat, savemat
    import tables

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
