import sys
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize('extension.pyx', language_level=sys.version_info[0])
)
