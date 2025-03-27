# sparseconverter
Format detection, identifiers and converter matrix for a range of numerical array formats (backends) in Python, focusing on sparse arrays.

## Usage

Basic usage:

```python
import numpy as np
import sparseconverter as spc

a1 = np.array([
    (1, 0, 3),
    (0, 0, 6)
])

# array conversion
a2 = spc.for_backend(a1, spc.SPARSE_GCXS)

# format determination
print("a1 is", spc.get_backend(a1), "and a2 is", spc.get_backend(a2))
```

```
a1 is numpy and a2 is sparse.GCXS
```


See `examples/` directory for more!

## Description

This library can help to implement algorithms that support a wide range of array formats as input, output or
for internal calculations. All dense and sparse array libraries already do support format detection, creation and export from and to various formats,
but with different APIs, different sets of formats and different sets of supported features -- dtypes, shapes, device classes etc.

This project creates an unified API for all conversions between the supported formats and takes care of details such as reshaping,
dtype conversion, and using an efficient intermediate format for multi-step conversions.

## Features
* Supports Python 3.9 - (at least) 3.13
* Defines constants for format identifiers
* Various sets to group formats into categories:
  * Dense vs sparse
  * CPU vs CuPy-based
  * nD vs 2D backends
* Efficiently detect format of arrays, including support for subclasses
* Get converter function for a pair of formats
* Convert to a target format
* Find most efficient conversion pair for a range of possible inputs and/or outputs

That way it can help to implement format-specific optimized versions of an algorithm,
to specify which formats are supported by a specific routine, to adapt to
availability of CuPy on a target machine,
and to perform efficient conversion to supported formats as needed.

## Supported array formats
* [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
* [`numpy.matrix`](https://numpy.org/doc/stable/reference/generated/numpy.matrix.html) -- to support result of aggregation operations on scipy.sparse matrices
* [`cupy.ndarray`](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html)
* [`sparse.COO`](https://sparse.pydata.org/en/stable/generated/sparse.COO.html)
* [`sparse.GCXS`](https://sparse.pydata.org/en/stable/generated/sparse.GCXS.html)
* [`sparse.DOK`](https://sparse.pydata.org/en/stable/generated/sparse.DOK.html)
* [`scipy.sparse.coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html)
* [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
* [`scipy.sparse.csc_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)
* [`scipy.sparse.coo_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html)
* [`scipy.sparse.csr_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html)
* [`scipy.sparse.csc_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html)
* [`cupyx.scipy.sparse.coo_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.coo_matrix.html)
* [`cupyx.scipy.sparse.csr_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csr_matrix.html)
* [`cupyx.scipy.sparse.csc_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csc_matrix.html)

## Still TODO

* PyTorch arrays
* More detailed cost metric based on more real-world use cases and parameters.

## Changelog

### 0.6.0 (in development)

* No changes yet

### 0.5.0

* Drop support for Python 3.8 https://github.com/LiberTEM/sparseconverter/pull/61
* Add support for Python 3.13 https://github.com/LiberTEM/sparseconverter/pull/61

### 0.4.0

* Better error message in case of unknown array type: https://github.com/LiberTEM/sparseconverter/pull/37
* Support for SciPy sparse arrays: https://github.com/LiberTEM/sparseconverter/pull/52
* Drop support for Python 3.7: https://github.com/LiberTEM/sparseconverter/pull/51

### 0.3.4

* Support for Python 3.12 https://github.com/LiberTEM/sparseconverter/pull/26
* Packaging update: Tests for conda-forge https://github.com/LiberTEM/sparseconverter/pull/27

### 0.3.3

* Perform feature checks lazily https://github.com/LiberTEM/sparseconverter/issues/15

### 0.3.2

* Detection and workaround for https://github.com/pydata/sparse/issues/602.
* Detection and workaround for https://github.com/cupy/cupy/issues/7713.
* Test with duplicates and scrambled indices.
* Test correctness of basic array operations.

### 0.3.1

* Include version constraint for `sparse`.

### 0.3.0

* Introduce `conversion_cost()` to obtain a value roughly proportional to the conversion cost
  between two backends.

### 0.2.0

* Introduce `result_type()` to find the smallest NumPy dtype that accomodates
  all parameters. Allowed as parameters are all valid arguments to
  `numpy.result_type(...)` plus backend specifiers.
* Support `cupyx.scipy.sparse.csr_matrix` with `dtype=bool`.

### 0.1.1

Initial release

## Known issues

* `conda install -c conda-forge cupy` on Python 3.7 and Windows 11 may install `cudatoolkit` 10.1 and `cupy` 8.3, which have sporadically produced invalid data structures for `cupyx.sparse.csc_matrix` for unknown reasons. This doesn't happen with current versions. Running the benchmark function `benchmark_conversions()` can help to debug such issues since it performs all pairwise conversions and checks for correctness.

## Notes

This project is developed primarily for sparse data support in [LiberTEM](https://libertem.github.io). For that reason it includes
the backend `CUDA`, which indicates a NumPy array, but targeting execution on a CUDA device.
