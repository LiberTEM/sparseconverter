# Advanced example:
# * Restrict set of formats to specific classes
# * Automatically selecting an efficient converter

import scipy.sparse

from sparseconverter import (
    CPU_BACKENDS, CUPY, NUMPY, SPARSE_COO, SPARSE_GCXS,
    cheapest_pair, for_backend, get_backend
)

# We want to support a range of multidimensional array formats
SUPPORTED = {NUMPY, SPARSE_COO, SPARSE_GCXS, CUPY}

# we detect if we have working CuPy support
try:
    import cupy

    a = cupy.array((1, 2, 3))
    a.sum()
except Exception:
    # CuPy can be a bit fragile and fail for various reasons
    cupy = None

# We only use CPU formats in case we don't have CuPy or it is broken
if cupy is None:
    SUPPORTED = SUPPORTED.intersection(CPU_BACKENDS)


# Our exemplary algorithm
def bin_last_axis(arr, bin_factor=2):
    '''
    Bin the last axis with the given bin factor.

    We require n-dimensional array support for this, but want to support
    sparse matrices as well.
    '''
    # We figure out which target format can be created efficiently from the
    # input data
    _, right = cheapest_pair((get_backend(arr), ), SUPPORTED)
    # We convert to the format of choice
    converted = for_backend(arr, right)

    # Binning from
    # https://stackoverflow.com/questions/21921178/binning-a-numpy-array
    selected = converted[..., :(converted.shape[-1] // bin_factor) * bin_factor]
    # (3, 7, 16) becomes (3, 7, -1, 2)
    reshaped = selected.reshape(selected.shape[:-1] + (-1, 2))
    return reshaped.mean(axis=-1)


if __name__ == '__main__':
    # Create CSR input data
    arr1 = scipy.sparse.eye(7, format='csr')
    res1 = bin_last_axis(arr1)
    # The return type is GCXS since it can be created efficiently from CSC or CSR
    print("Example 1")
    print(res1, get_backend(res1))

    # Create COO input data
    arr2 = scipy.sparse.eye(7, format='coo')
    res2 = bin_last_axis(arr2)
    # The return type is COO since conversion from scipy.sparse.coo_matrix to
    # sparse.COO is efficient
    print("Example 2")
    print(res2, get_backend(res2))
