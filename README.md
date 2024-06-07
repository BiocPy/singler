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
data = sce.read_tenx_h5("pbmc4k-tenx.h5", realize_assays=True)
mat = data.assay("counts")
features = [str(x) for x in data.row_data["name"]]
```

or if you are coming from scverse ecosystem, i.e. `AnnData`, simply read the object as `SingleCellExperiment` and extract the matrix and the features.
Read more on [SingleCellExperiment here](https://biocpy.github.io/tutorial/chapters/experiments/single_cell_experiment.html).


```python
import singlecellexperiment as sce

sce_adata = sce.SingleCellExperiment.from_anndata(adata) 

# or from a h5ad file
sce_h5ad = sce.read_h5ad("tests/data/adata.h5ad")
```

Now, we fetch the Blueprint/ENCODE reference:

```python
import celldex

ref_data = celldex.fetch_reference("blueprint_encode", "2024-02-26", realize_assays=True)
```

We can annotate each cell in `mat` with the reference:

```python
import singler
results = singler.annotate_single(
    test_data = mat,
    test_features = features,
    ref_data = ref_data,
    ref_labels = "label.main",
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

```python
built = singler.build_single_reference(
    ref_data=ref_data.assay("logcounts"),
    ref_labels=ref_data.col_data.column("label.main"),
    ref_features=ref_data.get_row_names(),
    restrict_to=features,
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

    ## output
    BiocFrame with 4340 rows and 3 columns
                best                                   scores                delta
            <list>                              <BiocFrame>   <ndarray[float64]>
    [0] Monocytes 0.33265560369962943:0.407117403330602...  0.40706830113982534
    [1] Monocytes 0.4078771641637374:0.4783396310685646...  0.07000418564184802
    [2] Monocytes 0.3517036021728629:0.4076971245524348...  0.30997293412307647
                ...                                      ...                  ...
    [4337]  NK cells 0.3472631136865701:0.3937898240670208...  0.09640242155786138
    [4338]   B-cells 0.26974632191999887:0.334862058137758... 0.061215905058676856
    [4339] Monocytes 0.39390119034537324:0.468867490667427...  0.06678168346812047

## Integrating labels across references

We can use annotations from multiple references through the `annotate_integrated()` function:

```python
import singler
import celldex

blueprint_ref = celldex.fetch_reference("blueprint_encode", "2024-02-26", realize_assays=True)

immune_cell_ref = celldex.fetch_reference("dice", "2024-02-26", realize_assays=True)

single_results, integrated = singler.annotate_integrated(
    mat,
    features,
    ref_data_list = (blueprint_ref, immune_cell_ref),
    ref_labels_list = "label.main",
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
