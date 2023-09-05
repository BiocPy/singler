<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/singler.svg?branch=main)](https://cirrus-ci.com/github/<USER>/singler)
[![ReadTheDocs](https://readthedocs.org/projects/singler/badge/?version=latest)](https://singler.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/singler/main.svg)](https://coveralls.io/r/<USER>/singler)
[![PyPI-Server](https://img.shields.io/pypi/v/singler.svg)](https://pypi.org/project/singler/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/singler.svg)](https://anaconda.org/conda-forge/singler)
[![Monthly Downloads](https://pepy.tech/badge/singler/month)](https://pepy.tech/project/singler)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/singler)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# Tinder for single-cell data

This package provides Python bindings to the [C++ implementation](https://github.com/LTLA/singlepp) of the [SingleR algorithm](https://github.com/LTLA/SingleR).
It is designed to annotate cell types by matching cells to known references based on their expression profiles.


<!-- pyscaffold-notes -->

## Developer notes

To rebuild the **ctypes** bindings [**cpptypes**](https://github.com/BiocPy/ctypes-wrapper):

```shell
cpptypes src/singler/lib --py src/singler/cpphelpers.py --cpp src/singler/lib/bindings.cpp
```

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.

