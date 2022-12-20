# Advanced example:
# * Restrict set of formats to specific classes
# * Automatically selecting an efficient converter

import scipy.sparse

import sparseconverter as spc

# We want to support a range of multidimensional array formats
SUPPORTED = frozenset({spc.NUMPY, spc.SPARSE_COO, spc.SPARSE_GCXS, spc.CUPY})

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
    SUPPORTED = SUPPORTED.intersection(spc.CPU_BACKENDS)


# Our exemplary algorithm
def bin_last_axis(arr, bin_factor=2):
    '''
    Bin the last axis with the given bin factor.

    We require n-dimensional array support for this, but want to support
    sparse matrices as well.
    '''
    # We figure out which target format can be created efficiently from the
    # input data
    _, right = spc.cheapest_pair((spc.get_backend(arr), ), SUPPORTED)
    # We convert to the format of choice
    converted = spc.for_backend(arr, right)

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
    print(res1, spc.get_backend(res1))

    # Create COO input data
    arr2 = scipy.sparse.eye(7, format='coo')
    res2 = bin_last_axis(arr2)
    # The return type is COO since conversion from scipy.sparse.coo_matrix to
    # sparse.COO is efficient
    print("Example 2")
    print(res2, spc.get_backend(res2))
