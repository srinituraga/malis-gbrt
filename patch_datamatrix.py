from __future__ import division
import numbers
import numpy as np
from itertools import product

# class PatchDataMatrix(np.ndarray):
#   """Extracts patches from an ndarray datacube"""
#   def __new__(cls, datacube, patchSz=None, stride=None):

class PatchDataMatrix:
    """Extracts patches from an ndarray datacube"""

    def __init__(self, datacube, patchSz=None, stride=None):
        self.datacube = datacube
        if patchSz == None:
            self.patchSz = (1,)*(self.datacube.ndim-1)
        else:
            self.patchSz = patchSz
        if stride == None:
            self.stride = (1,)*(self.datacube.ndim-1)
        else:
            self.stride = stride

        self.shapeOutput = tuple([(i-j+1)/k for i,j,k in zip(self.datacube.shape[:-1],self.patchSz,self.stride)])
        self.ndim = 2
        self.shape = (np.prod(self.shapeOutput), self.datacube.shape[-1]*np.prod(self.patchSz))
        self.size = np.prod(self.shape)
        self.dtype = self.datacube.dtype

    # def __getitem__(self, key):
    #   if  

def extract_patches_3d(image,
                patch_size,
                patch_stride = (1,1,1),
                max_patches=None,
                random_state=None,
                order="F",
                dtype=None):
    """Reshape a 3D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Parameters
    ----------
    image: array, shape = (image_height, image_width, image_depth) or
        (image_height, image_width, image_depth, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size: tuple of ints (patch_height, patch_width, patch_depth)
        the dimensions of one patch

    max_patches: integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.

    Returns
    -------
    patches: array, shape = (n_patches, patch_height, patch_width, patch_depth) or
         (n_patches, patch_height, patch_width, patch_depth, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.

    # Examples
    # --------

    # >>> from sklearn.feature_extraction import image
    # >>> one_image = np.arange(16).reshape((4, 4))
    # >>> one_image
    # array([[ 0,  1,  2,  3],
    #        [ 4,  5,  6,  7],
    #        [ 8,  9, 10, 11],
    #        [12, 13, 14, 15]])
    # >>> patches = image.extract_patches_2d(one_image, (2, 2))
    # >>> print patches.shape
    # (9, 2, 2)
    # >>> patches[0]
    # array([[0, 1],
    #        [4, 5]])
    # >>> patches[1]
    # array([[1, 2],
    #        [5, 6]])
    # >>> patches[8]
    # array([[10, 11],
    #        [14, 15]])
    """
    i_h, i_w, i_d = image.shape[:3]
    p_h, p_w, p_d = patch_size
    s_h, s_w, s_d = patch_stride

    if not len(image.shape) == 4:
        image = image.reshape((i_h, i_w, i_d, -1))
    n_colors = image.shape[-1]

    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_d = i_d - p_d + 1
    all_patches = n_h * n_w * n_d
    n_dim = np.prod((np.ceil(p_h/s_h), np.ceil(p_w/s_w), np.ceil(p_d/s_d), n_colors))

    if dtype==None:
        dtype = image.dtype

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            n_patches = max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            n_patches = int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)

        rng = check_random_state(random_state)
        patches = np.empty((n_patches, n_dim), dtype=dtype, order=order)
        i_s = rng.randint(n_h, size=n_patches)
        j_s = rng.randint(n_w, size=n_patches)
        k_s = rng.randint(n_d, size=n_patches)
        for p, i, j, k in zip(patches, i_s, j_s, k_s):
            # print image[i:i + p_h:s_h, j:j + p_w:s_w, k:k + p_d:s_d, :].shape
            # print image[i:i + p_h:s_h, j:j + p_w:s_w, k:k + p_d:s_d, :].ravel().shape
            # print p[:].shape
            p[:] = image[i:i + p_h:s_h, j:j + p_w:s_w, k:k + p_d:s_d, :].ravel()
    else:
        n_patches = all_patches
        patches = np.empty((n_patches, n_dim), dtype=dtype, order=order)
        for p, (i, j, k) in zip(patches, product(xrange(n_h), xrange(n_w), xrange(n_d))):
            p[:] = image[i:i + p_h:s_h, j:j + p_w:s_w, k:k + p_d:s_d, :].ravel()

    return patches

def extract_patches_3d_affinity(seg,
                patch_size,
                patch_idx_keep,
                max_patches=None,
                random_state=None,
                order="F",
                dtype=None):

    i_h, i_w, i_d = seg.shape[:3]
    p_h, p_w, p_d = patch_size

    assert(len(seg.shape) == 3)

    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_d = i_d - p_d + 1
    all_patches = n_h * n_w * n_d
    n_dim = np.prod([len(l) for l in patch_idx_keep])
    patch_center = n_dim/2

    if dtype==None:
        dtype = seg.dtype

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            n_patches = max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            n_patches = int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)

        rng = check_random_state(random_state)
        patches = np.empty((n_patches, n_dim), dtype=dtype, order=order)
        i_s = rng.randint(n_h, size=n_patches)
        j_s = rng.randint(n_w, size=n_patches)
        k_s = rng.randint(n_d, size=n_patches)
        for p, i, j, k in zip(patches, i_s, j_s, k_s):
            pp = seg[np.ix_(    [i+idx for idx in patch_idx_keep[0]],
                                [j+idx for idx in patch_idx_keep[1]],
                                [k+idx for idx in patch_idx_keep[2]]    )].ravel()
            p[:] = np.logical_and(pp==pp[patch_center],pp[patch_center]>0)
    else:
        n_patches = all_patches
        patches = np.empty((n_patches, n_dim), dtype=dtype, order=order)
        for p, (i, j, k) in zip(patches, product(xrange(n_h), xrange(n_w), xrange(n_d))):
            # pp = seg[i:i + p_h:s_h, j:j + p_w:s_w, k:k + p_d:s_d].ravel()
            pp = seg[np.ix_(    [i+idx for idx in patch_idx_keep[0]],
                                [j+idx for idx in patch_idx_keep[1]],
                                [k+idx for idx in patch_idx_keep[2]]    )].ravel()
            p[:] = np.logical_and(pp==pp[patch_center],pp[patch_center]>0)
            # p[:] = (pp==pp[int(pp.size/2)]) and (pp[int(pp.size/2)] > 0)

    return patches

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


 

# Tests...
if __name__ == "__main__":
    a = np.random.randn(256,256,256,11)
    b = PatchDataMatrix(datacube=a,patchSz=(25,25,25))
    c = PatchDataMatrix(datacube=a,patchSz=(25,25,25),stride=(5,5,5))

    # import sklearn
    # from sklearn.feature_extraction.image import extract_patches_2d,extract_patches_3d
    # d = extract_patches_3d(a,(25,25,25))
    # d.shape = (d.shape[0],-1)
    e = extract_patches_3d(a,(21,21,21),(5,5,5),max_patches=0.5)
    e.shape = (e.shape[0],-1)
