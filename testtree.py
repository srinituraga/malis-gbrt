import numpy as np
import vigra
import tree

imShape = (15,20,25)
n_feat = 22
n_output = 3
max_depth = 4
min_samples_leaf = 100
patch_size = (51,51,51)
max_feat = int(np.ceil(np.sqrt(n_feat*np.prod(patch_size))))

n_node = 2**(max_depth+1)

x_im = np.zeros(imShape+(n_feat,),dtype='float32')
y_im = np.zeros(imShape+(n_output,),dtype='float32')
yValue = np.zeros((n_node,n_output),dtype='float32')
splitValue = np.zeros(n_node,dtype='float32')
splitFeat = np.zeros((n_node,4),dtype='int32')
childrenLeft = np.zeros(n_node,dtype='int32')
childrenRight = np.zeros(n_node,dtype='int32')

x_im[0,0,0,0] = 1.0
x_im[1,2,3,4] = 5.0
print '[python] x_im[0,0,0,0] =' , x_im[0,0,0,0]
print '[python] x_im[1,2,3,4] =' , x_im[1,2,3,4]
tree.fit(
			x_im,
			y_im,
			yValue,
			splitValue,
			splitFeat,
			childrenLeft,
			childrenRight,
			max_feat,
			max_depth,
			min_samples_leaf,
			patch_size[0], patch_size[1], patch_size[2]
		)
