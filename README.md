<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/singler.svg?branch=main)](https://cirrus-ci.com/github/<USER>/singler)
[![ReadTheDocs](https://readthedocs.org/projects/singler/badge/?version=latest)](https://singler.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/singler/main.svg)](https://coveralls.io/r/<USER>/singler)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/singler.svg)](https://anaconda.org/conda-forge/singler)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/singler)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/singler.svg)](https://pypi.org/project/singler/)
[![Monthly Downloads](https://static.pepy.tech/badge/singler/month)](https://pepy.tech/project/singler)
![Unit tests](https://github.com/BiocPy/singler/actions/workflows/pypi-test.yml/badge.svg)

# Tinder for single-cell data

## Overview

This package provides Python bindings to the [C++ implementation](https://github.com/LTLA/singlepp) of the [SingleR algorithm](https://github.com/LTLA/SingleR),
originally developed by [Aran et al. (2019)](https://www.nature.com/articles/s41590-018-0276-y).
It is designed to annotate cell types by matching cells to known references based on their expression profiles.
So kind of like Tinder, but for cells.

## Quick start

Firstly, let's load in the famous PBMC 4k dataset from 10X Genomics:

```python
import singlecellexperiment as sce
data = sce.read_tenx_h5("pbmc4k-tenx.h5")
mat = data.assay("counts")
features = [str(x) for x in data.row_data["name"]]
```

Now we use the Blueprint/ENCODE reference to annotate each cell in `mat`:

```python
import singler
results = singler.annotate_single(
    mat,
    features,
    ref_data = "BlueprintEncode",
    ref_features = "symbol",
    ref_labels = "main",
    cache_dir = "_cache"
)
```

The `results` data frame contains all of the assignments and the scores for each label:

```python
results.column("best")
## ['Monocytes',
##  'Monocytes',
##  'Monocytes',
##  'CD8+ T-cells',
##  'CD4+ T-cells',
##  'CD8+ T-cells',
##  'Monocytes',
##  'Monocytes',
##  'B-cells',
##  ...
## ]

results.column("scores").column("Macrophages")
## array([0.35935275, 0.40833545, 0.37430726, ..., 0.32135929, 0.29728435,
##        0.40208581])
```

## Calling low-level functions

The `annotate_single()` function is a convenient wrapper around a number of lower-level functions in **singler**.
Advanced users may prefer to build the reference and run the classification separately.
This allows us to re-use the same reference for multiple datasets without repeating the build step.

We start by fetching the reference of interest from [GitHub](https://github.com/kanaverse/singlepp-references).
Note the use of `cache_dir` to avoid repeated downloads from GitHub.

```python
ref = singler.fetch_github_reference("BlueprintEncode", cache_dir="_cache")
```

We'll be using the gene symbols here with the markers for the main labels.
We need to set `restrict_to` to the features in our test data, so as to avoid picking marker genes in the reference that won't be present in the test.

```python
ref_features = ref.row_data.column("symbol")

markers = singler.realize_github_markers(
    ref.metadata["main"],
    ref_features,
    restrict_to=set(features),
)
```

Now we build the reference from the ranked expression values and the associated labels in the reference:

```python
built = singler.build_single_reference(
    ref_data=ref.assay("ranks"),
    ref_labels=ref.col_data.column("main"),
    ref_features=ref_features,
    markers=markers,
)
```

And finally, we apply the pre-built reference to the test dataset to obtain our label assignments.
This can be repeated with different datasets that have the same features or a superset of `features`.

```python
output = singler.classify_single_reference(
    mat,
    test_features=features,
    ref_prebuilt=built,
)
```

## Integrating labels across references

We can use annotations from multiple references through the `annotate_integrated()` function:

```python
import singler
single_results, integrated = singler.annotate_integrated(
    mat,
    features,
    ref_data = ("BlueprintEncode", "DatabaseImmuneCellExpression"),
    ref_features = "symbol",
    ref_labels = "main",
    build_integrated_args = { "ref_names": ("Blueprint", "DICE") },
    cache_dir = "_cache",
    num_threads = 6
)
```

This annotates the test dataset against each reference individually to obtain the best per-reference label,
and then it compares across references to find the best label from all references.
Both the single and integrated annotations are reported for diagnostics.

```python
integrated.column("best_label")
## ['Monocytes', 
##  'Monocytes',
##  'Monocytes',
##  'CD8+ T-cells',
##  'CD4+ T-cells',
##  'CD8+ T-cells',
##  'Monocytes',
##  'Monocytes',
##  ...
## ]

integrated.column("best_reference")
## ['Blueprint',
## 'Blueprint',
## 'Blueprint',
## 'Blueprint',
## 'Blueprint',
## 'Blueprint',
## 'Blueprint',
## ...
##]
```

## Developer notes

Build the shared object file:

```shell
python setup.py build_ext --inplace
```

For quick testing:

```shell
pytest
```

For more complex testing:

```shell
python setup.py build_ext --inplace && tox
```

To rebuild the **ctypes** bindings with [**cpptypes**](https://github.com/BiocPy/ctypes-wrapper):

```shell
cpptypes src/singler/lib --py src/singler/_cpphelpers.py --cpp src/singler/lib/bindings.cpp --dll _core
```
