import os
from setuptools import setup

install_requires = [
    "numpy >= 1.18",
    "pandas >= 1.0",
    "joblib >= 1.0",
    "duckdb >= 0.4",
    "pypika >= 0.48",
    "pyarrow >= 8.0.0",
]

extras_require = {
    "base": [],
    "test": [
        "pytest >= 6.0.0",
        "scikit-learn >= 1.0.0",  # sklearn 1.0+ requires python 3.7+
        "ipython",
        "mongomock",
    ],
}

extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

packages = [
    "mandala_lite",
    "mandala_lite.core",
    "mandala_lite.queries",
    "mandala_lite.storages",
    "mandala_lite.tests",
    "mandala_lite.ui",
]

setup(
    name="mandala_lite",
    version="0.1.0",
    description="",
    url="https://github.com/amakelov/mandala_lite",
    license="Apache 2.0",
    keywords="computational-experiments data-management machine-learning data-science",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=packages,
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
)
