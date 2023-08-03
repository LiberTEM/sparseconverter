import random

import pytest
import numpy as np
import sparse
from numpy.testing import assert_allclose

from sparseconverter import (
    CPU_BACKENDS, CUDA, CUPY_BACKENDS, CUPY_SCIPY_COO, CUPY_SCIPY_CSC, CUPY_SCIPY_CSR,
    DENSE_BACKENDS, NUMPY, BACKENDS, CUDA_BACKENDS, ND_BACKENDS, SCIPY_COO, SCIPY_CSC,
    SCIPY_CSR, SPARSE_BACKENDS, SPARSE_COO, SPARSE_DOK, SPARSE_GCXS,
    cheapest_pair, check_shape, for_backend, get_backend, get_device_class, make_like,
    prod, benchmark_conversions, result_type, conversion_cost
)


# we detect if we have working CuPy support
try:
    import cupy

    a = cupy.array((1, 2, 3))
    a.sum()
except Exception:
    # CuPy can be a bit fragile and fail for various reasons
    cupy = None


# This function introduces asymmetries so that errors won't average out so
# easily with large data sets
# Copied from LiberTEM tests
def _mk_random(size, dtype='float32', array_backend=NUMPY):
    size = tuple(size)
    if array_backend not in ND_BACKENDS and len(size) != 2:
        raise ValueError(f"Format {array_backend} does not support size {size}")
    dtype = np.dtype(dtype)
    if array_backend in SPARSE_BACKENDS:
        if array_backend == SPARSE_GCXS:
            form = 'gcxs'
        else:
            form = 'coo'
        data = for_backend(sparse.random(size, format=form).astype(dtype), array_backend)
    elif array_backend in DENSE_BACKENDS:
        if dtype.kind == 'c':
            choice = [0, 2, -4, 0+5j, 0-6j]
        else:
            choice = [0, 3]
        data = np.random.choice(choice, size=size).astype(dtype)
        coords2 = tuple(np.random.choice(range(c)) for c in size)
        coords10 = tuple(np.random.choice(range(c)) for c in size)
        data[coords2] = np.random.choice(choice) * sum(size)
        data[coords10] = np.random.choice(choice) * 10 * sum(size)
        data = for_backend(data, array_backend)
    else:
        raise ValueError(f"Don't understand array format {array_backend}.")
    if data.dtype != dtype:
        raise ValueError(f"Can't make array with format {array_backend} and dtype {dtype}.")
    if data.shape != size:
        raise ValueError(
            f"Can't make array with format {array_backend} and shape {size}, "
            f"got shape {data.shape}."
        )
    if array_backend in (SCIPY_COO, CUPY_SCIPY_COO):
        data = _scramble_coo(_add_duplicate_coo(data))
    elif array_backend in (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC, SCIPY_CSR, SCIPY_CSC):
        data = _scramble_csr_csc(_add_duplicate_csc_csr(data))
    return data


def _add_duplicate_csc_csr(arr):
    if arr.nnz > 0:
        if arr.dtype in (bool, np.uint8, np.uint16, np.uint32, np.uint64):
            new_value = 0
        else:
            new_value = 1
        new_data = np.append(arr.data, (0, new_value))
        if new_value:
            new_data[-3] -= new_value
        last_index = arr.indices[-1]
        new_indices = np.append(arr.indices, (last_index, last_index))
        new_indptr = arr.indptr.copy()
        for last_not_empty in range(1, len(new_indptr)):
            if new_indptr[-last_not_empty] != new_indptr[-last_not_empty - 1]:
                break
        for i in range(1, last_not_empty + 1):
            new_indptr[-i] += 2
        arr = arr.__class__(
            (new_data, new_indices, new_indptr),
            shape=arr.shape,
            dtype=arr.dtype
        )
    return arr


def _scramble_csr_csc(arr):
    # We build an array with compound type to shuffle data and indices together
    # even though they have different dtypes
    entry_dtype = [
        ('data', arr.data.dtype),
        ('indices', arr.indices.dtype),
    ]
    new_data = arr.data.copy()
    new_indices = arr.indices.copy()
    for i in range(len(arr.indptr) - 1):
        start = int(arr.indptr[i])
        stop = int(arr.indptr[i+1])
        sl = np.empty(shape=stop-start, dtype=entry_dtype)
        sl['data'] = for_backend(arr.data[start:stop], get_backend(sl['data']))
        sl['indices'] = for_backend(arr.indices[start:stop], get_backend(sl['indices']))
        np.random.shuffle(sl)
        new_data[start:stop] = for_backend(sl['data'], get_backend(arr.data))
        new_indices[start:stop] = for_backend(sl['indices'], get_backend(arr.indices))
    arr = arr.__class__(
        (new_data, new_indices, arr.indptr),
        shape=arr.shape,
        dtype=arr.dtype
    )
    assert not arr.has_sorted_indices
    return arr


def _add_duplicate_coo(arr):
    if arr.nnz > 0:
        if arr.dtype in (bool, np.uint8, np.uint16, np.uint32, np.uint64):
            new_value = 0
        else:
            new_value = 1
        new_data = np.append(arr.data, (0, new_value))
        if new_value:
            new_data[-3] -= new_value
        last_col = arr.col[-1]
        new_col = np.append(arr.col, (last_col, last_col))
        last_row = arr.row[-1]
        new_row = np.append(arr.row, (last_row, last_row))
        return arr.__class__((new_data, (new_row, new_col)), shape=arr.shape, dtype=arr.dtype)
    else:
        return arr


def _scramble_coo(arr):
    # We build an array with compound type to shuffle data and indices together
    # even though they have different dtypes
    entry_dtype = [
        ('data', arr.data.dtype),
        ('row', arr.row.dtype),
        ('col', arr.col.dtype),
    ]
    sl = np.empty(len(arr.data), dtype=entry_dtype)
    sl['data'] = for_backend(arr.data, get_backend(sl['data']))
    sl['row'] = for_backend(arr.row, get_backend(sl['row']))
    sl['col'] = for_backend(arr.col, get_backend(sl['col']))
    np.random.shuffle(sl)
    new_data = for_backend(sl['data'], get_backend(arr.data))
    new_row = for_backend(sl['row'], get_backend(arr.row))
    new_col = for_backend(sl['col'], get_backend(arr.col))

    return arr.__class__((new_data, (new_row, new_col)), shape=arr.shape, dtype=arr.dtype)


@pytest.mark.parametrize(
    'left', BACKENDS
)
@pytest.mark.parametrize(
    'right', BACKENDS
)
@pytest.mark.parametrize(
    'dtype', [
        bool, float, int,
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
        np.float32, np.float64, np.complex64, np.complex128,
    ]
)
@pytest.mark.parametrize(
    'scramble', (True, False)
)
def test_for_backend(left, right, dtype, scramble):
    CUPY_SPARSE_DTYPES = {
        np.float32, np.float64, np.complex64, np.complex128
    }
    CUPY_SPARSE_CSR_DTYPES = {
        bool, np.float32, np.float64, np.complex64, np.complex128
    }
    CUPY_SPARSE_FORMATS = {
        CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
    }
    print(left, right, dtype)

    if cupy is None and (left in CUDA_BACKENDS or right in CUDA_BACKENDS):
        pytest.skip("No CuPy, skipping CuPy test")
    if left == CUPY_SCIPY_CSR and dtype in CUPY_SPARSE_CSR_DTYPES:
        pass
    elif left in CUPY_SPARSE_FORMATS and dtype not in CUPY_SPARSE_DTYPES:
        pytest.skip(f"Dtype {dtype} not supported for left format {left}, skipping.")
    shape = (7, 11, 13, 17)

    def aggregate(arr):
        backend = get_backend(arr)
        # Doesn't support aggregation
        if backend in (SCIPY_CSR, CUPY_SCIPY_CSR) and arr.dtype == bool:
            return None
        # Doesn't support slicing
        elif backend in (SCIPY_COO, CUPY_SCIPY_COO):
            return None
        try:
            return for_backend(arr[2:4].sum(axis=0), NUMPY).reshape(shape[1:])
        except NotImplementedError:
            pass
        return None

    left_ref = _mk_random(shape, dtype=dtype, array_backend=NUMPY)
    left_ref_agg = aggregate(left_ref)
    assert isinstance(left_ref, np.ndarray)
    data = for_backend(left_ref, left)
    # On CUDA 10.1 and CuPy 8.3 one may end up with invalid data structures for
    # CSC matrices that will error out here.
    # That's what `conda install -c conda-forge cupy` installed on Python 3.7
    # and Windows 11 at the time of writing.
    if hasattr(data, 'toarray'):
        data.toarray()

    assert_allclose(for_backend(data, NUMPY).reshape(left_ref.shape), left_ref)

    if scramble:
        if left in (SCIPY_COO, CUPY_SCIPY_COO):
            data = _scramble_coo(_add_duplicate_coo(data))
        elif left in (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC, SCIPY_CSR, SCIPY_CSC):
            data = _scramble_csr_csc(_add_duplicate_csc_csr(data))
        else:
            pytest.skip(f"Scrambling not supported for backend {left}.")

    if left == CUDA:
        assert get_backend(data) == NUMPY
    else:
        assert get_backend(data) == left

    assert_allclose(for_backend(data, NUMPY).reshape(left_ref.shape), left_ref)

    left_agg = aggregate(data)

    # See above!
    converted = for_backend(data, right)
    if hasattr(converted, 'toarray'):
        converted.toarray()

    if right == CUDA:
        assert get_backend(converted) == NUMPY
    else:
        assert get_backend(converted) == right

    converted_agg = aggregate(converted)

    converted_back = for_backend(converted, NUMPY)

    back_agg = aggregate(converted_back)

    assert isinstance(converted_back, np.ndarray)

    if left in ND_BACKENDS and right in ND_BACKENDS:
        target_shape = shape
    else:
        target_shape = (shape[0], prod(shape[1:]))

    if left in ND_BACKENDS:
        check_shape(converted, shape)
    else:
        check_shape(converted, target_shape)

    assert converted.shape == target_shape
    assert converted_back.shape == target_shape
    try:
        assert_allclose(left_ref.reshape(target_shape), converted_back)
    except AssertionError:
        import sys
        opt = np.get_printoptions()
        np.set_printoptions(threshold=sys.maxsize)
        left_ref_reshaped = left_ref.reshape(target_shape)
        print("left_ref", repr(left_ref_reshaped))
        print("converted_back", repr(converted_back))
        print("difference", converted_back - left_ref_reshaped)
        print("data", repr(data))
        print("converted", repr(converted))
        np.set_printoptions(**opt)
        raise

    if left_agg is not None:
        assert_allclose(left_ref_agg, left_agg)
    if converted_agg is not None:
        assert_allclose(left_ref_agg, converted_agg)
    if back_agg is not None:
        assert_allclose(left_ref_agg, back_agg)

    keep_dtype = left not in CUPY_SPARSE_FORMATS and right not in CUPY_SPARSE_FORMATS
    keep_dtype = keep_dtype or dtype in CUPY_SPARSE_DTYPES
    no_bool = (CUPY_SCIPY_COO, CUPY_SCIPY_CSC)
    keep_dtype = keep_dtype or dtype == bool and left not in no_bool and right not in no_bool

    target_dtype = result_type(left, right, dtype)

    if (keep_dtype):
        assert left_ref.dtype == data.dtype
        assert left_ref.dtype == converted.dtype
        assert left_ref.dtype == converted_back.dtype
    else:
        assert converted.dtype == target_dtype


def test_unknown_format():
    not_an_array = "I am not a supported array type"
    res = get_backend(not_an_array)
    assert res is None

    # Not strict returns identity for an unknown array type or backend
    assert for_backend(not_an_array, 'asdf', strict=False) is not_an_array

    with pytest.raises(ValueError):
        for_backend(not_an_array, 'asdf', strict=True)


def test_cheapest_pair():
    source_backends = (NUMPY, SPARSE_DOK)
    target_backends = (NUMPY, SPARSE_GCXS)

    # Identity is cheapest
    assert cheapest_pair(source_backends, target_backends) == (NUMPY, NUMPY)


def test_device_class():
    for b in CPU_BACKENDS:
        assert get_device_class(b) == 'cpu'
    for b in CUPY_BACKENDS:
        assert get_device_class(b) == 'cupy'
    for b in (CUDA, ):
        assert get_device_class(b) == 'cuda'


@pytest.mark.parametrize(
    'source_array', [
        _mk_random((3, 4, 5), array_backend=NUMPY),
        _mk_random((3, 4), array_backend=SCIPY_COO),
    ]
)
@pytest.mark.parametrize(
    'target', [SPARSE_COO, SCIPY_COO]
)
def test_make_like(source_array, target):
    b = for_backend(source_array, target)
    c = make_like(b, source_array)

    assert c.shape == source_array.shape
    assert np.allclose(
        for_backend(c, NUMPY),
        for_backend(source_array, NUMPY)
    )
    # With strict=False we accept any reshape that matches in size
    numpy_target = np.empty((prod(source_array.shape),))
    d = make_like(b, numpy_target, strict=False)
    assert d.shape == numpy_target.shape
    assert np.allclose(
        d,
        for_backend(source_array, NUMPY).reshape((-1, ))
    )
    # With strict=True we expect a 100 % match for ND arrays
    # For 2D arrays we expect the first axis to be equal and the second
    # axis of the 2D array to be the product of the other second axes
    with pytest.raises(ValueError):
        make_like(b, numpy_target, strict=True)


def test_graceful_no_cupy():
    if cupy is not None:
        pytest.skip("FIXME mock missing CuPy")
    with pytest.raises((ModuleNotFoundError, ImportError)):
        benchmark_conversions(
            shape=(3, 5, 7),
            dtype=float,
            density=0.1,
            backends=(NUMPY, CUPY_SCIPY_COO),
            repeats=1,
            warmup=False
        )
    benchmark_conversions(
        shape=(3, 5, 7),
        dtype=float,
        density=0.1,
        backends=CPU_BACKENDS,
        repeats=1,
        warmup=False
    )


@pytest.mark.parametrize(
    'args,expected', [
        ((bool, ), bool),
        ((np.uint8, ), np.uint8),
        ((bool, CUPY_SCIPY_COO), np.float32),
        ((np.uint8, CUPY_SCIPY_COO), np.float32),
        ((np.uint8, CUPY_SCIPY_CSR, NUMPY), np.float32),
        ((np.complex64, CUPY_SCIPY_COO, np.int16, NUMPY), np.complex64),
        ((np.uint32, NUMPY, CUPY_SCIPY_COO, bool), np.float64),
        ((np.uint64, NUMPY, bool), np.uint64),
        ((CUPY_SCIPY_CSC, np.uint64, NUMPY, bool), np.float64),
        (DENSE_BACKENDS.union(
                SPARSE_BACKENDS.intersection(CPU_BACKENDS), (CUPY_SCIPY_CSR, )
            ), bool),
        (BACKENDS, np.float32),
    ]
)
def test_result_type(args, expected):
    print(args, expected)
    res = result_type(*args)
    assert res == expected


def test_conversion_cost():
    backend_l = list(BACKENDS)
    for i in range(10):
        left = random.choice(backend_l)
        right = random.choice(backend_l)

        print("cost between", left, right)
        cost = conversion_cost(left, right)
        print("cost", cost)
        assert isinstance(cost, float)
        assert np.isfinite(cost)
