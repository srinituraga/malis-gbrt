from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = 'Volume decision tree',
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension("_tree", ["_tree.pyx"])]
)
