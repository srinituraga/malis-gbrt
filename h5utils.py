import h5py
import time
import numpy
import os, sys

#*******************************************************************************
# H 5 P a t h                                                                  *
#*******************************************************************************

class H5Path(object):
    def __init__(self, p):
        l = p.find(".h5")
        self.filename  = p[0:l+3]
        self.groupname = p[l+4:]

def to5Dshape(data3d):
    assert len(data3d.shape) == 3
    return data3d.reshape((1,) + data3d.shape + (1,))

def rH5data(fname, dname=None, slicing=None, verbose=False):
    """read in a single dataset from a given HDF5 file"""
    start = time.time()
    if not os.path.exists(fname):
        raise RuntimeError("file '%s' does not exist" % fname)
    try:
        f = h5py.File(fname, 'r')
    except:
        raise RuntimeError("could not open file '%s'" % fname)
    if dname is None:
        g = f
        dname = ''
        while(True):
            if len(g.keys()) != 1:
                raise RuntimeError("no dataset name given, and I cannot determine"
                                   "the correct dataset to load for you")
            dname = dname + "/" + g.keys()[0]
            if type(f[dname]) == h5py._hl.group.Group:
                g = f[dname]
            else:
                break
    if verbose:
        sys.stdout.write("reading %s/%s" % (fname, dname))
        sys.stdout.flush()
    d = None
    if(slicing is not None):
        d = f[dname][slicing]
    else:
        d = f[dname].value
    f.close()
    if verbose:
        sys.stdout.write(" (shape=%r, dtype=%s) [%.3f sec]\n" \
               % (d.shape,d.dtype,time.time()-start))
    return d

def rH5shape(fname, dname):
    if not os.path.exists(fname):
        raise RuntimeError("file '%s' does not exist" % fname)
    try:
        f = h5py.File(fname, 'r')
    except:
        raise RuntimeError("could not open file '%s'" % fname)
    s = f[dname].shape
    f.close()
    return s

def wH5data(fname, dname, data, reverseShapeAttr=False, verbose=False, overwrite=False, slicing=None, chunkCompress=False):
    """write a single dataset to an HDF5 file"""
    f = None
    if slicing is not None:
        f = h5py.File(fname, 'a')
    else:
        if not overwrite and os.path.exists(fname):
            raise RuntimeError("path '%s' already exists" % fname)
        try:
            f = h5py.File(fname, 'w')
        except:
            raise RuntimeError("could not open file '%s' for writing" % fname)
    paths = dname.split("/")
    ds = f
    for i,path in enumerate(paths):
        if i != len(paths)-1:
            ds = ds.create_group(path)
        else:
            if slicing is None:
                if chunkCompress:
                    ds = ds.create_dataset(path, data=data, chunks=True, compression='gzip')
                else:
                    ds = ds.create_dataset(path, data=data)
                if(reverseShapeAttr):
                    ds.attrs.create("reverse-shape", "1")
            else:
                ds = ds[path]
                ds[slicing] = data

    if verbose:
        print "wrote '%s/%s' (shape=%r, dtype=%s)" % (fname,dname,data.shape,data.dtype)
    f.close()

#*******************************************************************************
# i f   _ _ n a m e _ _   = =   " _ _ m a i n _ _ "                            *
#*******************************************************************************

if __name__ == "__main__":
    a = numpy.zeros((2,2))
    a[1,0] = 42
    wH5data("/tmp/h5utils_test.h5", "some/long/path/a", a)
    b = rH5data("/tmp/h5utils_test.h5", "some/long/path/a")
    assert (a == b).all()
    del b

    wH5data("/tmp/h5utils_test.h5", "some/long/path/a", a, verbose=True)
    b = rH5data("/tmp/h5utils_test.h5", "some/long/path/a", verbose=True)
    assert (a == b).all()
