# sparseconverter
Converter matrix for a range of array formats (backends) in Python, focusing on sparse arrays.

This library is targeted at projects that want to support a wide range of array formats as input, output or
for calculations. All array libraries already do support format detection, creation and export from and to various formats,
but with different APIs, different sets of formats and different sets of supported features -- dtypes, shapes, device classes etc.

As an example, efficient conversion from `sparse.COO` to `cupyx.scipy.sparse.coo_matrix` can be done via `cupyx.scipy.sparse.coo_matrix(sparse.COO.to_scipy_sparse())`.
However, both `scipy.sparse.coo_matrix` and `cupyx.scipy.sparse.coo_matrix` only support 2D arrays. On top of that, `cupyx.scipy.sparse.coo_matrix`
only supports floating point dtypes and `bool`.

This project creates an unified API for all conversions between the supported formats and takes care of details such as using an efficient intermediate format, reshaping and dtype conversion.

## Features
* Supports Python 3.6 - 3.10
* Defines constants for format identifiers
* Various sets to group formats into categories:
  * Dense vs sparse
  * CPU vs CuPy-based
  * nD vs 2D backends
* Efficiently detect format of arrays, including support for subclasses
* Get converter function for a pair of formats
* Convert to a target format
* Find most efficient conversion pair for a range of possible inputs and/or outputs

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
* [`cupyx.scipy.sparse.coo_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.coo_matrix.html)
* [`cupyx.scipy.sparse.csr_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csr_matrix.html)
* [`cupyx.scipy.sparse.csc_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csc_matrix.html)

## Still TODO

* cupyx.sparse formats with dtype `bool`
* PyTorch arrays
* SciPy sparse arrays as opposed to SciPy sparse matrices.

## Notes

This project is developed primarily for sparse data support in [LiberTEM](https://libertem.github.io). For that reason it includes
the backend `CUDA`, which indicates a NumPy array, but targeting execution on a CUDA device.
