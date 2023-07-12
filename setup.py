import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

base_dir = os.path.dirname(os.path.abspath(__file__))

setup(name='torch_warper',
      ext_modules=[
          CUDAExtension('torch_warper', ['src/warp.cpp', 'src/warping.cu'],
                        include_dirs=[os.path.join(base_dir, 'include')],
                        extra_compile_args={
                            'cxx': ['-Wfatal-errors', '-O3', '-std=c++17'],
                            'nvcc': ['-O3', '-Xptxas', '-O3,-v']
                        }),
      ],
      cmdclass={'build_ext': BuildExtension})
