#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <queue>
#include <map>
#include <boost/python/tuple.hpp>
#include <boost/python.hpp>

using namespace std;

#define TREE_LEAF -1

/*
 * Fit a randomized decision tree
 */

boost::python::tuple fit(
    vigra::NumpyArray<4, float> x_im,
    vigra::NumpyArray<4, float> y_im,
    vigra::NumpyArray<2,float> yValue,
    vigra::NumpyArray<1,float> splitValue,
    vigra::NumpyArray<2,int> splitFeat,
    vigra::NumpyArray<1,int> childrenLeft,
    vigra::NumpyArray<1,int> childrenRight,
    int maxFeat,
    int maxDepth,
    int minSamplesLeaf,
    int patchSzi, int patchSzj, int patchSzk
) {
    
	/* input arrays */
    vigra::NumpyArray<4, float>::size_type x_dims = x_im.shape();
    const size_t nFeat = x_dims[3];
    // const float* x_data = (const float*)x_im.data();
    vigra::NumpyArray<4, float>::size_type y_dims = y_im.shape();
    const size_t nOutput = y_dims[1];
    // const float* y_data = (const float*)y_im.data();

    // cout << "[c++]    x_im(0,0,0,0) = " << x_im(0,0,0,0) << endl;
    // cout << "[c++]    x_im[0,0,0,0] = " << x_im[0,0,0,0] << endl;
    // cout << "[c++]    x_im(1,2,3,4) = " << x_im(1,2,3,4) << endl;
    // cout << "[c++]    x_im(4,3,2,1) = " << x_im(4,3,2,1) << endl;
    // cout << "[c++]    x_im(0,1,2,3) = " << x_im(0,1,2,3) << endl;
    // cout << "[c++]    x_im(3,2,1,0) = " << x_im(3,2,1,0) << endl;
    // cout << "maxDepth: " << maxDepth << endl;
    // cout << "maxNode: " << pow(2.0,maxDepth) << endl;

    


    /* return outputs */
    return boost::python::make_tuple(yValue,
                                    splitValue,
                                    splitFeat,
                                    childrenLeft,
                                    childrenRight);
     
}

BOOST_PYTHON_MODULE_INIT(tree) {
    vigra::import_vigranumpy();
    boost::python::def("fit", 
        vigra::registerConverters(&fit)
    );
}
