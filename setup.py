"""
Compile the python source code to c. 

EXCEPTIONS: list the exception files. 

"""

import os
from distutils.core import setup, Extension 
from Cython.Build import cythonize
import numpy

'''
EXCEPTIONS = ['setup.py', 'server.py', 'client.py']

files = []

for _, _, filenames in os.walk('./'):
    for onefile in filenames:
        files.append(onefile)

for item in files:
    if not item.endswith('.py'):
        EXCEPTIONS.append(item)
    if item.startswith('.'):
        EXCEPTIONS.append(item)

print('----------------------------------')
print('Exception Files: ')
print('----------------------------------')
for item in EXCEPTIONS:
    if item in files:
        print('EXCEPTION: ', item)
        files.remove(item)
print('----------------------------------')
'''
files = ['ws3.py', ]

compiler_directives = {'optimize.unpack_method_calls': False}

#extensions = [Extension(files, include_dirs = [numpy.get_include()])]

setup(ext_modules = cythonize(files, 
                        nthreads = 8, 
                        language_level = 3,
                        compiler_directives = compiler_directives, 
                        ))

#setup(ext_modules = cythonize(files, 
#            language = 'c',
#            include_dirs = [numpy.get_include()]
#            )))

