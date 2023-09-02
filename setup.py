from setuptools import setup
try:
  import cupy
except Exception:
  raise RuntimeError('CuPy is not available. Please install it manually')

setup(
    name='BrainDiffusion',
    version='0.1.0',
    description='Python package for anatomic connectivity of brain',
    author=' ',
    author_email=' ',
    license='BSD 2-clause',
    packages=['BrainDiffusion'],
    install_requires=['nibabel>=5.0.1',
                      'numpy>=1.23.5',
                      'pandas>=1.5.2',
                      'joblib>=1.1.1',
                      'scipy>=1.10.0',
                      'gdown',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)
