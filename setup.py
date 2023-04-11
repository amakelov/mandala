import os
from setuptools import setup

install_requires = [
    "numpy >= 1.18",
    "pandas >= 1.0",
    "joblib >= 1.0",
    "pypika >= 0.48",
    "pyarrow >= 8.0.0",
]

extras_require = {
    "base": [],
    "performance": [
        "cityhash >= 0.2.2",  # for faster content hashing
    ],
    "integrations": [
        "dask[complete]",
    ],
    "ui": [
        "rich",
    ],
    "test": [
        "pytest >= 6.0.0",
        "hypothesis >= 6.0.0",
        "ipython",
        "mongomock",
        "duckdb >= 0.6",
    ],
    "demos": [
        "torch",
        "scikit-learn",
    ],
    "server": [
        "pymongo",
    ],
}


extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

packages = [
    "mandala",
    "mandala.core",
    "mandala.deps",
    "mandala.deps.tracers",
    "mandala.queries",
    "mandala.storages",
    "mandala.storages.rel_impls",
    "mandala.storages.remote_impls",
    "mandala.ui",
    "mandala.tests",
]

setup(
    name="mandala",
    version="0.1.0",
    description="",
    url="https://github.com/amakelov/mandala",
    license="Apache 2.0",
    keywords="computational-experiments data-management machine-learning data-science",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
    ],
    packages=packages,
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
)
