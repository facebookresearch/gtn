#!/usr/bin/env python3

import os 
import platform 
import re 
import subprocess 
import sys 
from packaging import version 
from setuptools import Extension, setup 
from setuptools.command.build_ext import build_ext 

class CMakeExtension(Extension) : 
    def __init__(self, name) : 
        Extension.__init__(self, name, sources =[]) 

class CMakeBuild(build_ext) : 
    def run(self) : 
        try : 
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = re.search(r"version\s*([\d.]+)", out.decode().lower()).group(1)
        if version.parse(cmake_version) < version.parse("3.5.1"):
            raise RuntimeError("CMake >= 3.5.1 is required to build gtn")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath("gtn")
        srcdir = os.path.abspath("../..")
        #required for auto - detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DPROJECT_SOURCE_DIR=' + srcdir,
                      '-DGTN_BUILD_PYTHON_BINDINGS=ON',
                      '-DGTN_BUILD_EXAMPLES=OFF',
                      '-DGTN_BUILD_BENCHMARKS=OFF',
                      '-DGTN_BUILD_TESTS=OFF'
                      ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            #cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            #if sys.maxsize > 2 * *32:
            #cmake_args += ['-A', 'x64']
            #build_args += ['--', '/m']
            raise RuntimeError("gtn doesn't support building on Windows yet")
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', srcdir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name="gtn",
    version="0.0.1",
    author="Vineel Pratap",
    author_email="vineelkpratap@fb.com",
    description="gtn bindings for python",
    long_description="",
    ext_modules=[
        CMakeExtension("gtn._graph"),
        CMakeExtension("gtn._autograd"),
        CMakeExtension("gtn._utils"),
        CMakeExtension("gtn._functions"),
        CMakeExtension("gtn._torch"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)