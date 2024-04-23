#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    # emjay modified ------
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    # original ------------
    # name="diff_gauss",
    # packages=['diff_gauss'],
    # ---------------------    
    version="1.0.5",
    ext_modules=[
        CUDAExtension(
            # emjay modified -------
            name="diff_gaussian_rasterization._C",
            # original -------------
            # name="diff_gauss._C",
            # -----------------------
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
