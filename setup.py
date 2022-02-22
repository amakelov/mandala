import os
from setuptools import setup

install_requires = [
    'numpy >= 1.18',
    'pandas >= 1.0',
    'sqlalchemy >= 1.4',
    'tqdm >= 4.0',
    'joblib >= 1.0',
    'networkx >= 2.5'
]

extras_require = {
    'base': [],
    'pretty': ['rich >= 11.0.0'],
    'psql': ['psycopg2 >= 2.9'],
    'dask': ['dask >= 2021.0.0'],
    'ray': ['ray >= 1.0.0'],
    'test': [
        'pytest >= 6.0.0',
        'scikit-learn >= 1.0.0', # sklearn 1.0+ requires python 3.7+
        'ipython'
    ],
    'docs': [
        'altair >= 4.0.0',
        'scikit-learn >= 1.0.0',
        # 'scipy' # is required by sklearn already
    ]
}

extras_require['complete'] = sorted({v for req in extras_require.values() for v in req})

packages = [
    'mandala',
    'mandala.adapters',
    'mandala.core',
    'mandala.migration',
    'mandala.queries',
    'mandala.storages',
    'mandala.storages.kv_impl',
    'mandala.storages.rel_impl',
    'mandala.tests',
    'mandala.ui',
    'mandala.util',
]

setup(
    name='mandala',
    version='0.1.0',    
    description='Code-less experiment management',
    url='https://github.com/amakelov/mandala',
    author='Aleksandar Makelov',
    author_email='aleksandar.makelov@gmail.com',
    license="Apache 2.0",
    keywords="computational-experiments data-management machine-learning data-science",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6', - pickle fails on typing
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    packages=packages,
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require=extras_require,
)