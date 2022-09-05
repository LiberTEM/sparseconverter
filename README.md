# sparseconverter
Converter matrix for a range of array formats (backends) in Python, focusing on sparse arrays.

This library can help to implement support for a wide range of array formats as input, output or
for calculations. All dense and sparse array libraries already do support format detection, creation and export from and to various formats,
but with different APIs, different sets of formats and different sets of supported features -- dtypes, shapes, device classes etc.

This project creates an unified API for all conversions between the supported formats and takes care of details such as reshaping,
dtype conversion, and using an efficient intermediate format for multi-step conversions.

As an example, efficient conversion from a dense CuPy array `arr` to `sparse.COO` can be done by `sparse.COO(cupyx.scipy.sparse.coo_matrix(arr).get())`.
However, both `scipy.sparse.coo_matrix` and `cupyx.scipy.sparse.coo_matrix` only support 2D arrays. On top of that, `cupyx.scipy.sparse.coo_matrix`
only supports floating point dtypes and `bool`. The conversion function in `sparseconverters` is consequently:

```python
def _CUPY_to_sparse_coo(arr: cupy.ndarray):
    if arr.dtype in CUPY_SPARSE_DTYPES:
        reshaped = arr.reshape((arr.shape[0], -1))
        intermediate = cupyx.scipy.sparse.coo_matrix(reshaped)
        return sparse.COO(intermediate.get()).reshape(arr.shape)
    else:
        intermediate = cupy.asnumpy(arr)
        return sparse.COO.from_numpy(intermediate)
```

`sparseconverters` provides such tested conversion functions between all supported formats,
including a rough cost metric based on benchmarks.

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
* More detailed cost metric based on more real-world use cases and parameters.

## Notes

This project is developed primarily for sparse data support in [LiberTEM](https://libertem.github.io). For that reason it includes
the backend `CUDA`, which indicates a NumPy array, but targeting execution on a CUDA device.
