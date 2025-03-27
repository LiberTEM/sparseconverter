from functools import partial, reduce, lru_cache
import itertools
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional, Tuple, Union, Hashable
from typing_extensions import Literal

import numpy as np
import scipy.sparse as sp
import sparse


__version__ = '0.5.0'

NUMPY = 'numpy'
NUMPY_MATRIX = 'numpy.matrix'
CUDA = 'cuda'
CUPY = 'cupy'
SPARSE_COO = 'sparse.COO'
SPARSE_GCXS = 'sparse.GCXS'
SPARSE_DOK = 'sparse.DOK'

SCIPY_COO = 'scipy.sparse.coo_matrix'
SCIPY_CSR = 'scipy.sparse.csr_matrix'
SCIPY_CSC = 'scipy.sparse.csc_matrix'
SCIPY_COO_ARRAY = 'scipy.sparse.coo_array'
SCIPY_CSR_ARRAY = 'scipy.sparse.csr_array'
SCIPY_CSC_ARRAY = 'scipy.sparse.csc_array'

CUPY_SCIPY_COO = 'cupyx.scipy.sparse.coo_matrix'
CUPY_SCIPY_CSR = 'cupyx.scipy.sparse.csr_matrix'
CUPY_SCIPY_CSC = 'cupyx.scipy.sparse.csc_matrix'

ArrayBackend = Literal[
    'numpy', 'numpy.matrix', 'cuda',
    'cupy',
    'sparse.COO', 'sparse.GCXS', 'sparse.DOK',
    'scipy.sparse.coo_matrix', 'scipy.sparse.csr_matrix', 'scipy.sparse.csc_matrix',
    'scipy.sparse.coo_array', 'scipy.sparse.csr_array', 'scipy.sparse.csc_array',
    'cupyx.scipy.sparse.coo_matrix', 'cupyx.scipy.sparse.csr_matrix',
    'cupyx.scipy.sparse.csc_matrix',
]

CPU_BACKENDS = frozenset((
    NUMPY, NUMPY_MATRIX, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK, SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
    SCIPY_COO_ARRAY, SCIPY_CSR_ARRAY, SCIPY_CSC_ARRAY
))
CUPY_BACKENDS = frozenset((CUPY, CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC))
# "on CUDA, but no CuPy" backend that receives NumPy arrays
CUDA_BACKENDS = CUPY_BACKENDS.union((CUDA, ))
# Backends that don't require CUPY
NOCUPY_BACKENDS = CPU_BACKENDS.union((CUDA, ))
BACKENDS = CPU_BACKENDS.union(CUDA_BACKENDS)
# Backends that support n-dimensional arrays as opposed to 2D-only
ND_BACKENDS = frozenset((NUMPY, CUDA, CUPY, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK))
# 2D backends
D2_BACKENDS = frozenset((
    NUMPY_MATRIX, SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
    SCIPY_COO_ARRAY, SCIPY_CSR_ARRAY, SCIPY_CSC_ARRAY,
    CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
))
# Array classes that are not ND and not 2D are thinkable, i.e. automatically assuming
# a new class falls in these two categories may be wrong.
# Keep this consistency check in case arrays are added
assert ND_BACKENDS.union(D2_BACKENDS) == BACKENDS

SPARSE_BACKENDS = frozenset((
    SPARSE_COO, SPARSE_GCXS, SPARSE_DOK,
    SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
    SCIPY_COO_ARRAY, SCIPY_CSR_ARRAY, SCIPY_CSC_ARRAY,
    CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
))
DENSE_BACKENDS = BACKENDS - SPARSE_BACKENDS

DeviceClass = Literal['cpu', 'cuda', 'cupy']

if TYPE_CHECKING:
    import cupy
    import cupyx

try:
    sp.coo_array
except ModuleNotFoundError:
    raise RuntimeError("scipy.sparse.coo_array not found, requires scipy>=1.8")

ArrayT = Union[
    np.ndarray, np.matrix,
    sparse.SparseArray,
    sp.coo_matrix, sp.csr_matrix, sp.csc_matrix,
    sp.coo_array, sp.csr_array, sp.csc_array,
    "cupy.ndarray",
    "cupyx.scipy.sparse.coo_matrix",
    "cupyx.scipy.sparse.csr_matrix",
    "cupyx.scipy.sparse.csc_matrix"
]

Converter = Callable[[ArrayT], ArrayT]


# NOTE: inherits from `ValueError` instead of `TypeError` for
# backwards-compatibility.
class UnknownBackendError(ValueError):
    pass


class _ClassDict:
    '''
    Base classes for array types. The CuPy classes are loaded
    dynamically so that everything works without installed CuPy.

    The type of an array is determined through testing :code:`isinstance(arr, cls)`
    with these classes. The result for a specific type is cached so that repeated
    lookups don't have to do a full scan.
    '''
    _classes = {
        NUMPY_MATRIX: np.matrix,
        NUMPY: np.ndarray,
        CUDA: np.ndarray,
        SPARSE_COO: sparse.COO,
        SPARSE_GCXS: sparse.GCXS,
        SPARSE_DOK: sparse.DOK,
        SCIPY_COO_ARRAY: sp.coo_array,
        SCIPY_CSR_ARRAY: sp.csr_array,
        SCIPY_CSC_ARRAY: sp.csc_array,
        SCIPY_COO: sp.coo_matrix,
        SCIPY_CSR: sp.csr_matrix,
        SCIPY_CSC: sp.csc_matrix,
    }

    def __getitem__(self, item):
        res = self._classes.get(item, None)
        if res is None:
            res = self._get_lazy(item)
        return res

    def _get_lazy(self, item):
        if item == CUPY:
            import cupy
            res = cupy.ndarray
        elif item in (CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC):
            import cupyx.scipy  # noqa: F401
            # This is safe because it is one of the constants above
            # that are known good.
            res = eval(item)
        else:
            raise KeyError(f'Unknown format {item}')
        self._classes[item] = res
        return res


_classes = _ClassDict()

# Used for dtype upconversion for CuPy sparse matrices.
# bool is a neutral element of np.result_type()
_base_dtypes = {
    NUMPY: bool,
    NUMPY_MATRIX: bool,
    CUDA: bool,
    CUPY: bool,
    SPARSE_COO: bool,
    SPARSE_GCXS: bool,
    SPARSE_DOK: bool,
    SCIPY_COO: bool,
    SCIPY_CSR: bool,
    SCIPY_CSC: bool,
    SCIPY_COO_ARRAY: bool,
    SCIPY_CSR_ARRAY: bool,
    SCIPY_CSC_ARRAY: bool,
    CUPY_SCIPY_COO: np.float32,
    CUPY_SCIPY_CSC: np.float32,
    CUPY_SCIPY_CSR: np.float32,
}

# Exceptions for the above matrix
_special_mappings = {
    (CUPY_SCIPY_CSR, np.dtype(bool)): bool,
}


@lru_cache(maxsize=None)
def _test_GCXS_supports_non_canonical():
    # Checks for https://github.com/pydata/sparse/issues/602
    data = np.array((2., 1., 3., 3., 1.))
    indices = np.array((1, 0, 0, 1, 1), dtype=int)
    indptr = np.array((0, 2, 5), dtype=int)
    ref = np.array(((1., 2.), (3., 4.)))

    csr_check = sp.csr_matrix((data, indices, indptr), shape=(2, 2))
    csr_works = (
        np.all(csr_check[:1].todense() == ref[:1])
        and np.all(csr_check[1:].todense() == ref[1:])
    )
    if not csr_works:
        raise RuntimeError(
            'scipy.sparse.csr_matrix gave unexpected result. '
            'This is a bug, please report at https://github.com/LiberTEM/sparseconverter/!'
        )
    try:
        gcxs_check = sparse.GCXS(csr_check)
        gcxs_works = (
            np.all(gcxs_check[:1].todense() == ref[:1])
            and np.all(gcxs_check[1:].todense() == ref[1:])
        )
        return gcxs_works
    # Maybe a first "bandaid" for GCXS is throwing an error? In that case we canonicalize
    # the same way as if the bug was present.
    except Exception:
        raise  # FIXME remove
        return False


def _convert_csc_csr_to_pydata_sparse(left, right):
    """
    Build conversion function from CSR/CSC (left) to COO/GCXS/DOK
    (right) which lazily feature-checks for non-canonical support.
    """
    def _do_convert(arr):
        if _test_GCXS_supports_non_canonical():
            return _classes[right].from_scipy_sparse(arr)
        else:
            return chain(
                _ensure_sorted_dedup,
                _classes[right].from_scipy_sparse,
            )(arr)
    return _do_convert


def result_type(*args) -> np.dtype:
    '''
    Find a dtype that fulfills the following properties:

    * Can contain :code:`numpy.result_type(...)` of all items in :code:`args`
      that are not a backend specifier.
    * Supported by all items in :code:`args` that are backend specifiers
    '''
    backends = []
    others = []

    for a in args:
        try:
            if a in BACKENDS:
                backends.append(a)
            else:
                others.append(a)
        except TypeError:
            others.append(a)
    if others:
        result_dtype = np.result_type(*others)
    else:
        result_dtype = np.dtype(bool)
    stable = False
    while not stable:
        prev = result_dtype
        for b in backends:
            typ = _special_mappings.get((b, result_dtype), _base_dtypes[b])
            result_dtype = np.result_type(result_dtype, typ)
        stable = prev == result_dtype
    return result_dtype


def prod(shape: Tuple) -> int:
    '''
    np.prod forced to np.int64 to prevent integer overflows
    '''
    # Force 64 bit since int on Windows is 32 bit, leading to possible
    # overflows of np.prod(shape)
    return int(np.prod(shape, dtype=np.int64))


def _flatsig(arr: ArrayT) -> ArrayT:
    '''
    Convert to 2D for formats that only support two dimensions.

    All dimensions except the first one are flattened.
    '''
    target_shape = (arr.shape[0], prod(arr.shape[1:]))
    # print('_flatsig', arr.shape, type(arr), target_shape)
    return arr.reshape(target_shape)


def _identity(arr: ArrayT) -> ArrayT:
    # print('_identity', arr.shape, type(arr))
    return arr


def _GCXS_to_coo(arr: sparse.GCXS) -> sp.coo_matrix:
    '''
    The return type of :meth:`sparse.GCXS.to_scipy_sparse`
    depends on the compressed axes.
    '''
    reshaped = arr.reshape((arr.shape[0], -1))
    return reshaped.to_scipy_sparse().asformat('coo')


def _GCXS_to_csr(arr: sparse.GCXS) -> sp.csr_matrix:
    '''
    The return type of :meth:`sparse.GCXS.to_scipy_sparse`
    depends on the compressed axes.
    '''
    reshaped = arr.reshape((arr.shape[0], -1))
    return reshaped.to_scipy_sparse().asformat('csr')


def _GCXS_to_csc(arr: sparse.GCXS) -> sp.csc_matrix:
    '''
    The return type of :meth:`sparse.GCXS.to_scipy_sparse`
    depends on the compressed axes.
    '''
    reshaped = arr.reshape((arr.shape[0], -1))
    return reshaped.to_scipy_sparse().asformat('csc')


_CSR_CSC_T = Union[sp.csr_matrix, sp.csc_matrix]


def _ensure_sorted_dedup(arr: _CSR_CSC_T) -> _CSR_CSC_T:
    # Ensure we operate on a copy since sum_duplicates() is in_place
    if arr.has_sorted_indices:
        result = arr.copy()
    else:
        # Use the method that returns a copy
        result = arr.sorted_indices()
    result.sum_duplicates()
    return result


def chain(*functions: Converter) -> Converter:
    '''
    Create a function G(x) = f3(f2(f1(x)))
    from functions (f1, f2, f3)
    '''
    assert len(functions) >= 1
    return reduce(
        lambda val, func: (lambda x: func(val(x))),
        functions[1:],
        functions[0]
    )


class _ConverterDict:
    '''
    Manage a dictionary that maps a tuple of array types (left, right) to a
    converter function.

    The elements that require CuPy support are added dynamically by
    :meth:`_populate_cupy` when any CUDA-based types are requested. That way it
    works also without CuPy.
    '''
    def __init__(self):
        self._built = False
        self._converters = {}

    def _insert(self, key, function):
        self._converters[key] = function

    def _complete(self, key, function):
        if key not in self._converters:
            self._insert(key, function)

    def _build_converters(self):
        try:
            for backend in BACKENDS:
                self._insert((backend, backend), _identity)
            # Both are NumPy arrays, distinguished for device selection
            for left in (NUMPY, CUDA):
                for right in (NUMPY, CUDA):
                    self._insert((left, right), _identity)
            self._insert((NUMPY, NUMPY_MATRIX), chain(_flatsig, np.asmatrix))
            self._insert((NUMPY_MATRIX, NUMPY), np.asarray)

            for left in NUMPY, CUDA:
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    self._complete((left, right), _classes[right].from_numpy)

            formatcodes = {
                SPARSE_COO: 'coo',
                SPARSE_GCXS: 'gcxs',
                SPARSE_DOK: 'dok',
            }
            for left in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    self._complete(
                        (left, right),
                        partial(_classes[left].asformat, format=formatcodes[right])
                    )

            for left in (NUMPY, CUDA, SCIPY_COO, SCIPY_CSR, SCIPY_CSC, SCIPY_COO_ARRAY,
                    SCIPY_CSR_ARRAY, SCIPY_CSC_ARRAY):
                for right in (SCIPY_COO, SCIPY_CSR, SCIPY_CSC, SCIPY_COO_ARRAY, SCIPY_CSR_ARRAY,
                        SCIPY_CSC_ARRAY):
                    self._complete((left, right), chain(_flatsig, _classes[right]))
            for left in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                for right in NUMPY, CUDA:
                    self._complete((left, right), _classes[left].todense)
            for left in (SCIPY_COO, SCIPY_CSR, SCIPY_CSC, SCIPY_COO_ARRAY, SCIPY_CSR_ARRAY,
                        SCIPY_CSC_ARRAY):
                for right in NUMPY, CUDA:
                    self._complete((left, right), _classes[left].toarray)
            for left in SCIPY_CSR, SCIPY_CSC, SCIPY_CSR_ARRAY, SCIPY_CSC_ARRAY:
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    self._complete(
                        (left, right),
                        _convert_csc_csr_to_pydata_sparse(left, right)
                    )
            for left in SCIPY_COO, :
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    self._complete((left, right), _classes[right].from_scipy_sparse)

            # SCIPY_{COO,CSR,CSC}_ARRAY will be added by _complete_scipy_array()
            # using SCIPY_{COO,CSR,CSC} as a proxy since sparse doesn't have
            # support for conversion to and from SciPy sparse arrays
            self._insert((SPARSE_COO, SCIPY_COO), chain(_flatsig, sparse.COO.to_scipy_sparse))
            self._insert((SPARSE_COO, SCIPY_CSR), chain(_flatsig, sparse.COO.tocsr))
            self._insert((SPARSE_COO, SCIPY_CSC), chain(_flatsig, sparse.COO.tocsc))
            self._insert((SPARSE_GCXS, SCIPY_COO), _GCXS_to_coo)
            self._insert((SPARSE_GCXS, SCIPY_CSR), _GCXS_to_csr)
            self._insert((SPARSE_GCXS, SCIPY_CSC), _GCXS_to_csc)

            for right in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
                self._complete(
                    (SPARSE_DOK, right),
                    chain(
                        self._converters[(SPARSE_DOK, SPARSE_COO)],
                        self._converters[(SPARSE_COO, right)]
                    )
                )
            # Make sure any missing scipy.sparse.*_array entries are populated
            # before the following step
            self._complete_scipy_array()

            # Insert the conversion to and from NUMPY_MATRIX through NUMPY as a
            # proxy for all backends that are available without CuPy, in
            # particular the SciPy array backends
            for left in NOCUPY_BACKENDS:
                proxy = NUMPY
                right = NUMPY_MATRIX
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._complete((left, right), chain(c1, c2))

            for right in NOCUPY_BACKENDS:
                proxy = NUMPY
                left = NUMPY_MATRIX
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._complete((left, right), chain(c1, c2))

            self._check_cpu()
        except Exception:
            self._converters = None
            raise
        self._built = True

    def _complete_scipy_array(self):
        '''
        Fill all missing entries for SCIPY_{COO,CSR,CSC}_ARRAY with
        a proxy through SCIPY_{COO,CSR,CSC}
        '''
        proxies = {
            SCIPY_COO_ARRAY: SCIPY_COO,
            SCIPY_CSR_ARRAY: SCIPY_CSR,
            SCIPY_CSC_ARRAY: SCIPY_CSC,
        }
        for left in proxies.keys():
            for right in BACKENDS:
                proxy = proxies[left]
                if (proxy, right) in self._converters:
                    self._complete(
                        (left, right),
                        chain(_classes[proxy], self._converters[(proxy, right)])
                    )
                # Reverse direction
                if (right, proxy) in self._converters:
                    self._complete(
                        (right, left),
                        chain(self._converters[(right, proxy)], _classes[left])
                    )

    def _populate_cupy(self):
        import cupy
        import cupyx.scipy

        def detect_csr_complex128_bug():
            # https://github.com/cupy/cupy/issues/7035
            a = cupy.array([
                (1, 1j),
                (1j, -1-1j)

            ]).astype(np.complex128)
            sp = cupyx.scipy.sparse.csr_matrix(a)
            return not np.allclose(a, sp.todense())

        def detect_coo_works():
            works = True
            a = cupy.array([
                (1, 1j),
                (1j, -1-1j)

            ]).astype(np.complex128)
            try:
                cupyx.scipy.sparse.coo_matrix(a)
            except Exception:
                works = False
            return works

        def fail_coo_complex128(arr):
            raise NotImplementedError(
                "Please upgrade CuPy and/or CUDA: "
                f"Constructing {CUPY_SCIPY_COO} from {CUPY} doesn't work (old CuPy version)"
                "and installation is affected by this bug: "
                "https://github.com/cupy/cupy/issues/7035 "
                "(old cuSPARSE version)."
            )

        def detect_bool_csr_bug():
            data = cupy.array((True, True, True, False, True))
            indices = cupy.array((0, 1, 0, 1, 1), dtype=int)
            indptr = cupy.array((0, 2, 5), dtype=int)

            c_csr = cupyx.scipy.sparse.csr_matrix((data, indices, indptr))
            return not np.all(c_csr.todense())

        has_csr_complex128_bug = detect_csr_complex128_bug()
        has_cupy_coo_construction = detect_coo_works()
        has_bool_csr_bug = detect_bool_csr_bug()

        CUPY_SPARSE_DTYPES = {
            np.float32, np.float64, np.complex64, np.complex128
        }

        CUPY_SPARSE_CSR_DTYPES = {
            np.dtype(bool), np.float32, np.float64, np.complex64, np.complex128
        }

        def _adjust_dtype_cupy_sparse(array_backend: ArrayBackend):
            '''
            return dtype upconversion function for the specified cupyx.scipy.sparse backend.
            '''
            if array_backend == CUPY_SCIPY_CSR:
                allowed = CUPY_SPARSE_CSR_DTYPES
            elif array_backend in (CUPY_SCIPY_COO, CUPY_SCIPY_CSC):
                allowed = CUPY_SPARSE_DTYPES
            else:
                allowed = BACKENDS

            def adjust(arr):
                if arr.dtype in allowed:
                    res = arr
                    # print('_adjust_dtype_cupy_sparse passthrough', arr.dtype, res.dtype)
                else:
                    # Base dtype is the same for all cupyx.scipy.sparse matrices
                    res = arr.astype(result_type(arr, array_backend))
                return res

            return adjust

        def _GCXS_to_cupy_coo(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            return cupyx.scipy.sparse.coo_matrix(reshaped.to_scipy_sparse())

        def _GCXS_to_cupy_csr(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            return cupyx.scipy.sparse.csr_matrix(reshaped.to_scipy_sparse())

        def _GCXS_to_cupy_csc(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            return cupyx.scipy.sparse.csc_matrix(reshaped.to_scipy_sparse())

        def _GCXS_to_cupy(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            # Avoid changing the compressed axes
            if arr.compressed_axes == (0, ):
                return cupyx.scipy.sparse.csr_matrix(reshaped.to_scipy_sparse()).get()
            elif arr.compressed_axes == (1, ):
                return cupyx.scipy.sparse.csc_matrix(reshaped.to_scipy_sparse()).get()
            else:
                raise RuntimeError('Unexpected compressed axes in GCXS')

        def _CUPY_to_scipy_coo(arr: cupy.ndarray):
            '''
            Use GPU for sparsification if dtype allows, otherwise CPU
            '''
            reshaped = arr.reshape((arr.shape[0], -1))
            if arr.dtype in CUPY_SPARSE_DTYPES:
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped)
                return intermediate.get()
            else:
                intermediate = cupy.asnumpy(reshaped)
                return sp.coo_matrix(intermediate)

        def _CUPY_to_scipy_csr(arr: cupy.ndarray):
            '''
            Use GPU for sparsification if dtype allows, otherwise CPU
            '''
            reshaped = arr.reshape((arr.shape[0], -1))
            if arr.dtype in CUPY_SPARSE_CSR_DTYPES:
                intermediate = cupyx.scipy.sparse.csr_matrix(reshaped)
                return intermediate.get()
            else:
                intermediate = cupy.asnumpy(reshaped)
                return sp.csr_matrix(intermediate)

        def _CUPY_to_scipy_csc(arr: cupy.ndarray):
            '''
            Use GPU for sparsification if dtype allows, otherwise CPU
            '''
            reshaped = arr.reshape((arr.shape[0], -1))
            if arr.dtype in CUPY_SPARSE_DTYPES:
                intermediate = cupyx.scipy.sparse.csc_matrix(reshaped)
                return intermediate.get()
            else:
                intermediate = cupy.asnumpy(reshaped)
                return sp.csc_matrix(intermediate)

        def _CUPY_to_sparse_coo(arr: cupy.ndarray):
            '''
            Use GPU for sparsification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped)
                return sparse.COO(intermediate.get()).reshape(arr.shape)
            else:
                intermediate = cupy.asnumpy(arr)
                return sparse.COO.from_numpy(intermediate)

        def _CUPY_to_sparse_gcxs(arr: cupy.ndarray):
            '''
            Use GPU for sparsification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_CSR_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.csr_matrix(reshaped)
                intermediate = intermediate.get()
                if not _test_GCXS_supports_non_canonical():
                    intermediate.sort_indices()
                    intermediate.sum_duplicates()
                return sparse.GCXS(intermediate).reshape(arr.shape)
            else:
                intermediate = cupy.asnumpy(arr)
                return sparse.GCXS.from_numpy(intermediate)

        def _CUPY_to_sparse_dok(arr: cupy.ndarray):
            '''
            Use GPU for sparsification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped)
                return sparse.DOK(intermediate.get()).reshape(arr.shape)
            else:
                intermediate = cupy.asnumpy(arr)
                return sparse.DOK.from_numpy(intermediate)

        _adjust_dtype_for_cupy_scipy_coo = _adjust_dtype_cupy_sparse(CUPY_SCIPY_COO)
        _adjust_dtype_for_cupy_scipy_csr = _adjust_dtype_cupy_sparse(CUPY_SCIPY_CSR)
        _adjust_dtype_for_cupy_scipy_csc = _adjust_dtype_cupy_sparse(CUPY_SCIPY_CSC)

        def _CUPY_to_cupy_scipy_coo(arr: cupy.array):
            if has_cupy_coo_construction:
                return cupyx.scipy.sparse.coo_matrix(
                    _adjust_dtype_for_cupy_scipy_coo(
                        _flatsig(arr)
                    )
                )
            elif not has_csr_complex128_bug or arr.dtype != np.complex128:
                return cupyx.scipy.sparse.coo_matrix(
                    cupyx.scipy.sparse.csr_matrix(
                        _adjust_dtype_for_cupy_scipy_coo(
                            _flatsig(arr)
                        )
                    )
                )
            else:
                fail_coo_complex128(arr)

        def _CUPY_to_cupy_scipy_csr(arr: cupy.array):
            if has_csr_complex128_bug and arr.dtype == np.complex128:
                if has_cupy_coo_construction:
                    return cupyx.scipy.sparse.csr_matrix(
                        cupyx.scipy.sparse.coo_matrix(
                            _flatsig(arr)
                        )
                    )
                else:
                    return fail_coo_complex128(arr)
            else:
                return cupyx.scipy.sparse.csr_matrix(
                    _adjust_dtype_for_cupy_scipy_csr(
                        _flatsig(arr)
                    )
                )

        def _CUPY_to_cupy_scipy_csc(arr: cupy.array):
            if has_csr_complex128_bug and arr.dtype == np.complex128:
                if has_cupy_coo_construction:
                    return cupyx.scipy.sparse.csc_matrix(
                        cupyx.scipy.sparse.coo_matrix(
                            _flatsig(arr)
                        )
                    )
                else:
                    return fail_coo_complex128(arr)
            else:
                return cupyx.scipy.sparse.csc_matrix(
                    _adjust_dtype_for_cupy_scipy_csc(
                        _flatsig(arr)
                    )
                )

        def _sparse_coo_to_CUPY(arr: sparse.COO):
            '''
            Use GPU for densification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped.to_scipy_sparse())
                return intermediate.toarray().reshape(arr.shape)
            else:
                intermediate = arr.todense()
                return cupy.array(intermediate)

        def _sparse_gcxs_to_CUPY(arr: sparse.GCXS):
            '''
            Use GPU for densification if dtype allows, otherwise CPU
            '''
            if arr.compressed_axes == (0, ) and arr.dtype in CUPY_SPARSE_CSR_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.csr_matrix(reshaped.to_scipy_sparse())
                return intermediate.toarray().reshape(arr.shape)
            elif arr.compressed_axes == (1, ) and arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.csc_matrix(reshaped.to_scipy_sparse())
                return intermediate.toarray().reshape(arr.shape)
            else:
                intermediate = arr.todense()
                return cupy.array(intermediate)

        def _sparse_dok_to_CUPY(arr: sparse.DOK):
            '''
            Use GPU for densification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped.to_coo().to_scipy_sparse())
                return intermediate.toarray().reshape(arr.shape)
            else:
                intermediate = arr.todense()
                return cupy.array(intermediate)

        def _scipy_csr_to_CUPY(arr: sp.csr_matrix):
            '''
            Use GPU for densification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_CSR_DTYPES:
                if arr.dtype == bool and has_bool_csr_bug:
                    # Take copy since duplicates are summed in-place
                    arr = arr.copy()
                    arr.sum_duplicates()
                intermediate = cupyx.scipy.sparse.csr_matrix(arr)
                return intermediate.toarray()
            else:
                intermediate = arr.toarray()
                return cupy.array(intermediate)

        def _scipy_coo_to_CUPY(arr: sp.coo_matrix):
            '''
            Use GPU for densification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_DTYPES:
                intermediate = cupyx.scipy.sparse.coo_matrix(arr)
                return intermediate.toarray()
            else:
                intermediate = arr.toarray()
                return cupy.array(intermediate)

        def _scipy_csc_to_CUPY(arr: sp.coo_matrix):
            '''
            Use GPU for densification if dtype allows, otherwise CPU
            '''
            if arr.dtype in CUPY_SPARSE_DTYPES:
                intermediate = cupyx.scipy.sparse.csc_matrix(arr)
                return intermediate.toarray()
            else:
                intermediate = arr.toarray()
                return cupy.array(intermediate)

        def _cupy_scipy_csr_to_scipy_coo(arr: cupyx.scipy.sparse.csr_matrix):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                return cupyx.scipy.sparse.coo_matrix(arr).get()
            else:
                return sp.coo_matrix(arr.get())

        def _cupy_scipy_csr_to_scipy_csc(arr: cupyx.scipy.sparse.csr_matrix):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                return cupyx.scipy.sparse.csc_matrix(arr).get()
            else:
                return sp.csc_matrix(arr.get())

        def _cupy_scipy_csr_to_CUPY(arr: cupyx.scipy.sparse.csr_matrix):
            # Mitigation for https://github.com/cupy/cupy/issues/7713
            if arr.dtype == bool and has_bool_csr_bug:
                if not arr.has_canonical_format:
                    # sum_duplicates() doesn't work for bool, so we deduplicate on the CPU
                    cpu_arr = arr.get()
                    cpu_arr.sum_duplicates()
                    return cupyx.scipy.sparse.csr_matrix(cpu_arr).toarray()
            # Fallthrough
            return arr.toarray()

        self._insert((NUMPY, CUPY), cupy.array)
        self._insert((CUDA, CUPY), cupy.array)
        self._insert((CUPY, NUMPY), cupy.asnumpy)
        self._insert((CUPY, CUDA), cupy.asnumpy)

        self._insert((CUPY_SCIPY_CSR, CUPY), _cupy_scipy_csr_to_CUPY)

        for left in CUPY_SCIPY_COO, CUPY_SCIPY_CSC:
            self._complete((left, CUPY), _classes[left].toarray)

        for (left, right) in [
                    (CUPY_SCIPY_COO, SCIPY_COO),
                    (CUPY_SCIPY_CSR, SCIPY_CSR),
                    (CUPY_SCIPY_CSC, SCIPY_CSC)
                ]:
            self._complete((left, right), _classes[left].get)

        # Accepted by constructor of target class
        for left in (SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
                CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC):
            for right in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
                if left in ND_BACKENDS:
                    self._complete(
                        (left, right),
                        chain(_flatsig, _adjust_dtype_cupy_sparse(right), _classes[right])
                    )
                else:
                    self._complete(
                        (left, right),
                        chain(_adjust_dtype_cupy_sparse(right), _classes[right])
                    )
        # Work around https://github.com/cupy/cupy/issues/7035
        # Otherwise CUPY -> {CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC}
        # could be handled by the block above.
        self._insert((CUPY, CUPY_SCIPY_COO), _CUPY_to_cupy_scipy_coo)
        self._insert((CUPY, CUPY_SCIPY_CSR), _CUPY_to_cupy_scipy_csr)
        self._insert((CUPY, CUPY_SCIPY_CSC), _CUPY_to_cupy_scipy_csc)
        for left in NUMPY, CUDA:
            for right in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
                c1 = self._converters[(left, CUPY)]
                c2 = self._converters[(CUPY, right)]
                self._complete((left, right), chain(c1, c2))
        self._insert(
            (SPARSE_GCXS, CUPY_SCIPY_COO),
            chain(_adjust_dtype_cupy_sparse(CUPY_SCIPY_COO), _GCXS_to_cupy_coo)
        )
        self._insert((
            SPARSE_GCXS, CUPY_SCIPY_CSR),
            chain(_adjust_dtype_cupy_sparse(CUPY_SCIPY_CSR), _GCXS_to_cupy_csr)
        )
        self._insert(
            (SPARSE_GCXS, CUPY_SCIPY_CSC),
            chain(_adjust_dtype_cupy_sparse(CUPY_SCIPY_CSC), _GCXS_to_cupy_csc)
        )
        self._insert((SPARSE_GCXS, CUPY), _GCXS_to_cupy)

        self._insert((CUPY, SCIPY_COO), _CUPY_to_scipy_coo)
        self._insert((CUPY, SCIPY_CSR), _CUPY_to_scipy_csr)
        self._insert((CUPY, SCIPY_CSC), _CUPY_to_scipy_csc)

        self._insert((CUPY, SPARSE_COO), _CUPY_to_sparse_coo)
        self._insert((CUPY, SPARSE_GCXS), _CUPY_to_sparse_gcxs)
        self._insert((CUPY, SPARSE_DOK), _CUPY_to_sparse_dok)

        self._insert((SPARSE_COO, CUPY), _sparse_coo_to_CUPY)
        self._insert((SPARSE_GCXS, CUPY), _sparse_gcxs_to_CUPY)
        self._insert((SPARSE_DOK, CUPY), _sparse_dok_to_CUPY)

        self._insert((SCIPY_COO, CUPY), _scipy_coo_to_CUPY)
        self._insert((SCIPY_CSR, CUPY), _scipy_csr_to_CUPY)
        self._insert((SCIPY_CSC, CUPY), _scipy_csc_to_CUPY)

        self._insert((CUPY_SCIPY_CSR, SCIPY_COO), _cupy_scipy_csr_to_scipy_coo)
        self._insert((CUPY_SCIPY_CSR, SCIPY_CSC), _cupy_scipy_csr_to_scipy_csc)

        proxies = [
            (CUPY_SCIPY_COO, SCIPY_COO, NUMPY),
            (CUPY_SCIPY_COO, SCIPY_COO, CUDA),
            (CUPY_SCIPY_COO, CUPY_SCIPY_CSR, SCIPY_CSR),
            (CUPY_SCIPY_COO, CUPY_SCIPY_CSC, SCIPY_CSC),
            (CUPY_SCIPY_COO, SCIPY_COO, SPARSE_COO),
            (CUPY_SCIPY_COO, SCIPY_CSR, SPARSE_GCXS),
            (CUPY_SCIPY_COO, SCIPY_COO, SPARSE_DOK),

            (CUPY_SCIPY_CSR, SCIPY_CSR, NUMPY),
            (CUPY_SCIPY_CSR, SCIPY_CSR, CUDA),

            (CUPY_SCIPY_CSR, SCIPY_COO, SPARSE_COO),
            (CUPY_SCIPY_CSR, SCIPY_CSR, SPARSE_GCXS),
            (CUPY_SCIPY_CSR, SCIPY_CSR, SPARSE_DOK),

            (CUPY_SCIPY_CSC, SCIPY_CSC, NUMPY),
            (CUPY_SCIPY_CSC, SCIPY_CSC, CUDA),
            (CUPY_SCIPY_CSC, CUPY_SCIPY_COO, SCIPY_COO),
            (CUPY_SCIPY_CSC, CUPY_SCIPY_CSR, SCIPY_CSR),
            (CUPY_SCIPY_CSC, CUPY_SCIPY_COO, SPARSE_COO),
            (CUPY_SCIPY_CSC, SCIPY_CSC, SPARSE_GCXS),
            (CUPY_SCIPY_CSC, SCIPY_CSC, SPARSE_DOK),

            (SPARSE_COO, SCIPY_COO, CUPY_SCIPY_COO),
            (SPARSE_COO, SCIPY_COO, CUPY_SCIPY_CSR),
            (SPARSE_COO, SCIPY_COO, CUPY_SCIPY_CSC),
            (SPARSE_COO, SCIPY_COO, CUPY),

            (SPARSE_DOK, SCIPY_COO, CUPY_SCIPY_COO),
            (SPARSE_DOK, SCIPY_COO, CUPY_SCIPY_CSR),
            (SPARSE_DOK, SCIPY_COO, CUPY_SCIPY_CSC),
            (SPARSE_DOK, SCIPY_COO, CUPY),
        ]
        for left, proxy, right in proxies:
            c1 = self._converters[(left, proxy)]
            c2 = self._converters[(proxy, right)]
            self._complete((left, right), chain(c1, c2))

        for left in BACKENDS:
            proxy = NUMPY
            right = NUMPY_MATRIX
            c1 = self._converters[(left, proxy)]
            c2 = self._converters[(proxy, right)]
            self._complete((left, right), chain(c1, c2))
        for right in BACKENDS:
            proxy = NUMPY
            left = NUMPY_MATRIX
            c1 = self._converters[(left, proxy)]
            c2 = self._converters[(proxy, right)]
            self._complete((left, right), chain(c1, c2))

        self._complete_scipy_array()
        self._check_all()

    def _check_cpu(self):
        available = NOCUPY_BACKENDS
        for left in available:
            for right in available:
                if (left, right) not in self._converters:
                    raise RuntimeError(f'Missing converter {left} -> {right}')

    def _check_all(self):
        for left in BACKENDS:
            for right in BACKENDS:
                if (left, right) not in self._converters:
                    raise RuntimeError(f'Missing converter {left} -> {right}')

    def __getitem__(self, item):
        if self._converters is None:
            raise RuntimeError('Building the converter matrix has failed previously, aborting.')
        if not self._built:
            self._build_converters()
        res = self._converters.get(item, False)
        if res is False:
            left, right = item
            if left in CUDA_BACKENDS or right in CUDA_BACKENDS:
                self._populate_cupy()
            return self._converters[item]
        else:
            return res

    def get(self, item, default):
        try:
            return self.__getitem__(item)
        except KeyError:
            return default


_converters = _ConverterDict()
# Measured with a run of benchmark_conversions on a "representative machine" aka my laptop
# (32, 512, 512), np.float32, density 0.1
# See scripts/benchmark.ipynb to generate this list
# FIXME improve cost basis based on real use cases.
_cost = {
    (CUDA, CUDA): 7.5171e-06,
    (CUDA, CUPY): 0.003012379,
    (CUDA, CUPY_SCIPY_COO): 0.0184034938,
    (CUDA, CUPY_SCIPY_CSC): 0.0182290106,
    (CUDA, CUPY_SCIPY_CSR): 0.0190728582,
    (CUDA, NUMPY): 6.0776999999999994e-06,
    (CUDA, NUMPY_MATRIX): 6.54175e-05,
    (CUDA, SCIPY_COO_ARRAY): 0.0579601699,
    (CUDA, SCIPY_COO): 0.0582163223,
    (CUDA, SCIPY_CSC_ARRAY): 0.0637776906,
    (CUDA, SCIPY_CSC): 0.0618334691,
    (CUDA, SCIPY_CSR_ARRAY): 0.0610823155,
    (CUDA, SCIPY_CSR): 0.0636498886,
    (CUDA, SPARSE_COO): 0.0455485339,
    (CUDA, SPARSE_DOK): 0.4505717499,
    (CUDA, SPARSE_GCXS): 0.0794772298,
    (CUPY, CUDA): 0.0244937611,
    (CUPY, CUPY): 6.3493e-06,
    (CUPY, CUPY_SCIPY_COO): 0.0055470127,
    (CUPY, CUPY_SCIPY_CSC): 0.0053766754,
    (CUPY, CUPY_SCIPY_CSR): 0.0064715206,
    (CUPY, NUMPY): 0.025194987600000003,
    (CUPY, NUMPY_MATRIX): 0.0241695871,
    (CUPY, SCIPY_COO_ARRAY): 0.073479747,
    (CUPY, SCIPY_COO): 0.0794718939,
    (CUPY, SCIPY_CSC_ARRAY): 0.0848719941,
    (CUPY, SCIPY_CSC): 0.07185332909999999,
    (CUPY, SCIPY_CSR_ARRAY): 0.083021544,
    (CUPY, SCIPY_CSR): 0.0769946336,
    (CUPY, SPARSE_COO): 0.0587740921,
    (CUPY, SPARSE_DOK): 0.4278624361,
    (CUPY, SPARSE_GCXS): 0.0938476323,
    (CUPY_SCIPY_COO, CUDA): 0.0149334255,
    (CUPY_SCIPY_COO, CUPY): 0.07946194420000001,
    (CUPY_SCIPY_COO, CUPY_SCIPY_COO): 6.6387e-06,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSC): 0.0283525575,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSR): 0.0279808686,
    (CUPY_SCIPY_COO, NUMPY): 0.0165474797,
    (CUPY_SCIPY_COO, NUMPY_MATRIX): 0.0142636361,
    (CUPY_SCIPY_COO, SCIPY_COO_ARRAY): 0.0047082594,
    (CUPY_SCIPY_COO, SCIPY_COO): 0.0041682071,
    (CUPY_SCIPY_COO, SCIPY_CSC_ARRAY): 0.0430879439,
    (CUPY_SCIPY_COO, SCIPY_CSC): 0.043028589,
    (CUPY_SCIPY_COO, SCIPY_CSR_ARRAY): 0.0423622227,
    (CUPY_SCIPY_COO, SCIPY_CSR): 0.0424363589,
    (CUPY_SCIPY_COO, SPARSE_COO): 0.0135201129,
    (CUPY_SCIPY_COO, SPARSE_DOK): 0.9864800137,
    (CUPY_SCIPY_COO, SPARSE_GCXS): 0.0437462399,
    (CUPY_SCIPY_CSC, CUDA): 0.013548258300000001,
    (CUPY_SCIPY_CSC, CUPY): 0.0433026477,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_COO): 0.0016377708,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSC): 6.6961999999999995e-06,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSR): 0.0012530621000000001,
    (CUPY_SCIPY_CSC, NUMPY): 0.013553263800000001,
    (CUPY_SCIPY_CSC, NUMPY_MATRIX): 0.0135523716,
    (CUPY_SCIPY_CSC, SCIPY_COO_ARRAY): 0.0062360602999999995,
    (CUPY_SCIPY_CSC, SCIPY_COO): 0.0057640022,
    (CUPY_SCIPY_CSC, SCIPY_CSC_ARRAY): 0.002904996,
    (CUPY_SCIPY_CSC, SCIPY_CSC): 0.0027770586,
    (CUPY_SCIPY_CSC, SCIPY_CSR_ARRAY): 0.0062221453,
    (CUPY_SCIPY_CSC, SCIPY_CSR): 0.0063261925,
    (CUPY_SCIPY_CSC, SPARSE_COO): 0.059586117,
    (CUPY_SCIPY_CSC, SPARSE_DOK): 1.0324449221,
    (CUPY_SCIPY_CSC, SPARSE_GCXS): 0.0089073558,
    (CUPY_SCIPY_CSR, CUDA): 0.0119660899,
    (CUPY_SCIPY_CSR, CUPY): 0.0395033881,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_COO): 0.0010899782,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC): 0.0017298745,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSR): 7.501e-06,
    (CUPY_SCIPY_CSR, NUMPY): 0.0121105256,
    (CUPY_SCIPY_CSR, NUMPY_MATRIX): 0.011377737,
    (CUPY_SCIPY_CSR, SCIPY_COO_ARRAY): 0.0047116685,
    (CUPY_SCIPY_CSR, SCIPY_COO): 0.004341531599999999,
    (CUPY_SCIPY_CSR, SCIPY_CSC_ARRAY): 0.010740005,
    (CUPY_SCIPY_CSR, SCIPY_CSC): 0.0109886494,
    (CUPY_SCIPY_CSR, SCIPY_CSR_ARRAY): 0.0023193238999999997,
    (CUPY_SCIPY_CSR, SCIPY_CSR): 0.0024884889,
    (CUPY_SCIPY_CSR, SPARSE_COO): 0.0137379023,
    (CUPY_SCIPY_CSR, SPARSE_DOK): 0.9747494917,
    (CUPY_SCIPY_CSR, SPARSE_GCXS): 0.0044915743,
    (NUMPY, CUDA): 6.9657e-06,
    (NUMPY, CUPY): 0.0041761556,
    (NUMPY, CUPY_SCIPY_COO): 0.0215664355,
    (NUMPY, CUPY_SCIPY_CSC): 0.020297968,
    (NUMPY, CUPY_SCIPY_CSR): 0.0208672987,
    (NUMPY, NUMPY): 1.25847e-05,
    (NUMPY, NUMPY_MATRIX): 6.42753e-05,
    (NUMPY, SCIPY_COO_ARRAY): 0.0558553669,
    (NUMPY, SCIPY_COO): 0.0574968045,
    (NUMPY, SCIPY_CSC_ARRAY): 0.1532236645,
    (NUMPY, SCIPY_CSC): 0.0610968695,
    (NUMPY, SCIPY_CSR_ARRAY): 0.1287459439,
    (NUMPY, SCIPY_CSR): 0.1250039457,
    (NUMPY, SPARSE_COO): 0.045353941700000004,
    (NUMPY, SPARSE_DOK): 0.9625011438,
    (NUMPY, SPARSE_GCXS): 0.1813846489,
    (NUMPY_MATRIX, CUDA): 1.08512e-05,
    (NUMPY_MATRIX, CUPY): 0.002629923,
    (NUMPY_MATRIX, CUPY_SCIPY_COO): 0.0186175098,
    (NUMPY_MATRIX, CUPY_SCIPY_CSC): 0.0181780245,
    (NUMPY_MATRIX, CUPY_SCIPY_CSR): 0.0192458317,
    (NUMPY_MATRIX, NUMPY): 8.2015e-06,
    (NUMPY_MATRIX, NUMPY_MATRIX): 5.817e-06,
    (NUMPY_MATRIX, SCIPY_COO_ARRAY): 0.051768290700000004,
    (NUMPY_MATRIX, SCIPY_COO): 0.057946066,
    (NUMPY_MATRIX, SCIPY_CSC_ARRAY): 0.061657815899999996,
    (NUMPY_MATRIX, SCIPY_CSC): 0.0615601376,
    (NUMPY_MATRIX, SCIPY_CSR_ARRAY): 0.0596809239,
    (NUMPY_MATRIX, SCIPY_CSR): 0.0573960566,
    (NUMPY_MATRIX, SPARSE_COO): 0.041347750100000004,
    (NUMPY_MATRIX, SPARSE_DOK): 0.36838216960000003,
    (NUMPY_MATRIX, SPARSE_GCXS): 0.0745540076,
    (SCIPY_COO_ARRAY, CUDA): 0.0092566073,
    (SCIPY_COO_ARRAY, CUPY): 0.0138081092,
    (SCIPY_COO_ARRAY, CUPY_SCIPY_COO): 0.0063548308,
    (SCIPY_COO_ARRAY, CUPY_SCIPY_CSC): 0.0130204386,
    (SCIPY_COO_ARRAY, CUPY_SCIPY_CSR): 0.0061083065,
    (SCIPY_COO_ARRAY, NUMPY): 0.009392439300000001,
    (SCIPY_COO_ARRAY, NUMPY_MATRIX): 0.009266632,
    (SCIPY_COO_ARRAY, SCIPY_COO_ARRAY): 5.7878e-06,
    (SCIPY_COO_ARRAY, SCIPY_COO): 0.0007370949000000001,
    (SCIPY_COO_ARRAY, SCIPY_CSC_ARRAY): 0.010224924199999999,
    (SCIPY_COO_ARRAY, SCIPY_CSC): 0.0101457985,
    (SCIPY_COO_ARRAY, SCIPY_CSR_ARRAY): 0.0034711157,
    (SCIPY_COO_ARRAY, SCIPY_CSR): 0.0033891172,
    (SCIPY_COO_ARRAY, SPARSE_COO): 0.0090769005,
    (SCIPY_COO_ARRAY, SPARSE_DOK): 0.8621572069,
    (SCIPY_COO_ARRAY, SPARSE_GCXS): 0.004483293,
    (SCIPY_COO, CUDA): 0.0103207199,
    (SCIPY_COO, CUPY): 0.0186250175,
    (SCIPY_COO, CUPY_SCIPY_COO): 0.005428946400000001,
    (SCIPY_COO, CUPY_SCIPY_CSC): 0.013035086,
    (SCIPY_COO, CUPY_SCIPY_CSR): 0.0061293133,
    (SCIPY_COO, NUMPY): 0.0119568763,
    (SCIPY_COO, NUMPY_MATRIX): 0.009245578800000001,
    (SCIPY_COO, SCIPY_COO_ARRAY): 0.0007174556,
    (SCIPY_COO, SCIPY_COO): 6.2393000000000006e-06,
    (SCIPY_COO, SCIPY_CSC_ARRAY): 0.0085197359,
    (SCIPY_COO, SCIPY_CSC): 0.008473294300000002,
    (SCIPY_COO, SCIPY_CSR_ARRAY): 0.0030558563999999997,
    (SCIPY_COO, SCIPY_CSR): 0.003139234,
    (SCIPY_COO, SPARSE_COO): 0.0005846863,
    (SCIPY_COO, SPARSE_DOK): 0.9844116932,
    (SCIPY_COO, SPARSE_GCXS): 0.0035844711,
    (SCIPY_CSC_ARRAY, CUDA): 0.011217620300000002,
    (SCIPY_CSC_ARRAY, CUPY): 0.0154791225,
    (SCIPY_CSC_ARRAY, CUPY_SCIPY_COO): 0.0082053114,
    (SCIPY_CSC_ARRAY, CUPY_SCIPY_CSC): 0.0018183973000000001,
    (SCIPY_CSC_ARRAY, CUPY_SCIPY_CSR): 0.0049893967,
    (SCIPY_CSC_ARRAY, NUMPY): 0.011183402,
    (SCIPY_CSC_ARRAY, NUMPY_MATRIX): 0.0107541753,
    (SCIPY_CSC_ARRAY, SCIPY_COO_ARRAY): 0.0043565162000000004,
    (SCIPY_CSC_ARRAY, SCIPY_COO): 0.004433595,
    (SCIPY_CSC_ARRAY, SCIPY_CSC_ARRAY): 6.9423000000000004e-06,
    (SCIPY_CSC_ARRAY, SCIPY_CSC): 9.529610000000001e-05,
    (SCIPY_CSC_ARRAY, SCIPY_CSR_ARRAY): 0.0031522534,
    (SCIPY_CSC_ARRAY, SCIPY_CSR): 0.0031667176,
    (SCIPY_CSC_ARRAY, SPARSE_COO): 0.062038132600000004,
    (SCIPY_CSC_ARRAY, SPARSE_DOK): 1.015525802,
    (SCIPY_CSC_ARRAY, SPARSE_GCXS): 0.0040979184,
    (SCIPY_CSC, CUDA): 0.011534241300000001,
    (SCIPY_CSC, CUPY): 0.0156262137,
    (SCIPY_CSC, CUPY_SCIPY_COO): 0.0090378586,
    (SCIPY_CSC, CUPY_SCIPY_CSC): 0.0019040021000000002,
    (SCIPY_CSC, CUPY_SCIPY_CSR): 0.0053779321,
    (SCIPY_CSC, NUMPY): 0.0110324767,
    (SCIPY_CSC, NUMPY_MATRIX): 0.0108893985,
    (SCIPY_CSC, SCIPY_COO_ARRAY): 0.0045132985,
    (SCIPY_CSC, SCIPY_COO): 0.0040353948,
    (SCIPY_CSC, SCIPY_CSC_ARRAY): 8.371560000000001e-05,
    (SCIPY_CSC, SCIPY_CSC): 8.0602e-06,
    (SCIPY_CSC, SCIPY_CSR_ARRAY): 0.0030602884,
    (SCIPY_CSC, SCIPY_CSR): 0.0032227122,
    (SCIPY_CSC, SPARSE_COO): 0.06575373329999999,
    (SCIPY_CSC, SPARSE_DOK): 1.0452571339,
    (SCIPY_CSC, SPARSE_GCXS): 0.0039713711,
    (SCIPY_CSR_ARRAY, CUDA): 0.010055290599999999,
    (SCIPY_CSR_ARRAY, CUPY): 0.013836726,
    (SCIPY_CSR_ARRAY, CUPY_SCIPY_COO): 0.0066527092,
    (SCIPY_CSR_ARRAY, CUPY_SCIPY_CSC): 0.0102599591,
    (SCIPY_CSR_ARRAY, CUPY_SCIPY_CSR): 0.0015033216000000002,
    (SCIPY_CSR_ARRAY, NUMPY): 0.0104146315,
    (SCIPY_CSR_ARRAY, NUMPY_MATRIX): 0.009966180199999999,
    (SCIPY_CSR_ARRAY, SCIPY_COO_ARRAY): 0.0022747264,
    (SCIPY_CSR_ARRAY, SCIPY_COO): 0.0023314574,
    (SCIPY_CSR_ARRAY, SCIPY_CSC_ARRAY): 0.0091144021,
    (SCIPY_CSR_ARRAY, SCIPY_CSC): 0.008727211300000001,
    (SCIPY_CSR_ARRAY, SCIPY_CSR_ARRAY): 5.982100000000001e-06,
    (SCIPY_CSR_ARRAY, SCIPY_CSR): 7.899219999999999e-05,
    (SCIPY_CSR_ARRAY, SPARSE_COO): 0.0109655223,
    (SCIPY_CSR_ARRAY, SPARSE_DOK): 0.9863876457999999,
    (SCIPY_CSR_ARRAY, SPARSE_GCXS): 0.0011355428,
    (SCIPY_CSR, CUDA): 0.010444222599999999,
    (SCIPY_CSR, CUPY): 0.013690546699999999,
    (SCIPY_CSR, CUPY_SCIPY_COO): 0.006346078,
    (SCIPY_CSR, CUPY_SCIPY_CSC): 0.010284524199999999,
    (SCIPY_CSR, CUPY_SCIPY_CSR): 0.0014241769,
    (SCIPY_CSR, NUMPY): 0.010177361199999999,
    (SCIPY_CSR, NUMPY_MATRIX): 0.0095298086,
    (SCIPY_CSR, SCIPY_COO_ARRAY): 0.0022762851,
    (SCIPY_CSR, SCIPY_COO): 0.0023303186000000003,
    (SCIPY_CSR, SCIPY_CSC_ARRAY): 0.0088454384,
    (SCIPY_CSR, SCIPY_CSC): 0.008686323099999999,
    (SCIPY_CSR, SCIPY_CSR_ARRAY): 9.1875e-05,
    (SCIPY_CSR, SCIPY_CSR): 7.5628e-06,
    (SCIPY_CSR, SPARSE_COO): 0.0111089296,
    (SCIPY_CSR, SPARSE_DOK): 0.9730975649,
    (SCIPY_CSR, SPARSE_GCXS): 0.0010572409,
    (SPARSE_COO, CUDA): 0.0104704481,
    (SPARSE_COO, CUPY): 0.012988447699999999,
    (SPARSE_COO, CUPY_SCIPY_COO): 0.0191515764,
    (SPARSE_COO, CUPY_SCIPY_CSC): 0.0258278178,
    (SPARSE_COO, CUPY_SCIPY_CSR): 0.019493141399999997,
    (SPARSE_COO, NUMPY): 0.0107948896,
    (SPARSE_COO, NUMPY_MATRIX): 0.0099127454,
    (SPARSE_COO, SCIPY_COO_ARRAY): 0.0147836632,
    (SPARSE_COO, SCIPY_COO): 0.0149971549,
    (SPARSE_COO, SCIPY_CSC_ARRAY): 0.0226448533,
    (SPARSE_COO, SCIPY_CSC): 0.022331307600000003,
    (SPARSE_COO, SCIPY_CSR_ARRAY): 0.0144703722,
    (SPARSE_COO, SCIPY_CSR): 0.0143914904,
    (SPARSE_COO, SPARSE_COO): 5.9858e-06,
    (SPARSE_COO, SPARSE_DOK): 1.043537326,
    (SPARSE_COO, SPARSE_GCXS): 0.0317931642,
    (SPARSE_DOK, CUDA): 0.0993131669,
    (SPARSE_DOK, CUPY): 0.1075001119,
    (SPARSE_DOK, CUPY_SCIPY_COO): 0.4645485558,
    (SPARSE_DOK, CUPY_SCIPY_CSC): 0.455237905,
    (SPARSE_DOK, CUPY_SCIPY_CSR): 0.4564994924,
    (SPARSE_DOK, NUMPY): 0.1065822135,
    (SPARSE_DOK, NUMPY_MATRIX): 0.1032885821,
    (SPARSE_DOK, SCIPY_COO_ARRAY): 0.4535356606,
    (SPARSE_DOK, SCIPY_COO): 0.435443074,
    (SPARSE_DOK, SCIPY_CSC_ARRAY): 0.4556851598,
    (SPARSE_DOK, SCIPY_CSC): 0.450087391,
    (SPARSE_DOK, SCIPY_CSR_ARRAY): 0.4447047416,
    (SPARSE_DOK, SCIPY_CSR): 0.4491634283,
    (SPARSE_DOK, SPARSE_COO): 0.44738739889999996,
    (SPARSE_DOK, SPARSE_DOK): 7.466399999999999e-06,
    (SPARSE_DOK, SPARSE_GCXS): 0.4720391414,
    (SPARSE_GCXS, CUDA): 0.039875997600000004,
    (SPARSE_GCXS, CUPY): 0.0409483515,
    (SPARSE_GCXS, CUPY_SCIPY_COO): 0.28427135389999997,
    (SPARSE_GCXS, CUPY_SCIPY_CSC): 0.2680890195,
    (SPARSE_GCXS, CUPY_SCIPY_CSR): 0.2691131959,
    (SPARSE_GCXS, NUMPY): 0.0393779316,
    (SPARSE_GCXS, NUMPY_MATRIX): 0.037859172100000005,
    (SPARSE_GCXS, SCIPY_COO_ARRAY): 0.185717073,
    (SPARSE_GCXS, SCIPY_COO): 0.1862870509,
    (SPARSE_GCXS, SCIPY_CSC_ARRAY): 0.181581747,
    (SPARSE_GCXS, SCIPY_CSC): 0.1827632265,
    (SPARSE_GCXS, SCIPY_CSR_ARRAY): 0.1875337769,
    (SPARSE_GCXS, SCIPY_CSR): 0.1828276045,
    (SPARSE_GCXS, SPARSE_COO): 0.026042433,
    (SPARSE_GCXS, SPARSE_DOK): 1.0563622404,
    (SPARSE_GCXS, SPARSE_GCXS): 6.7738e-06,
}

# In order to support subclasses and not check all formats each time
# we cache dynamically which type maps to which format code
_type_cache: Dict[type, Optional[ArrayBackend]] = {}


def get_backend(arr: ArrayT) -> Optional[ArrayBackend]:
    '''
    Return the backend identifier for the given array

    Return :code:`None` if not identified.
    '''
    t = type(arr)
    backend = _type_cache.get(t, False)
    if backend is False:
        backend = None
        # Make sure to check NumPy matrix first since numpy.matrix is a subclass
        # of numpy.ndarray
        # Also check scipy sparse array first since in some versions the array is
        # a subclass of the corresponding matrix
        for b in ((
                NUMPY_MATRIX, SCIPY_COO_ARRAY, SCIPY_CSC_ARRAY,
                SCIPY_CSR_ARRAY) + tuple(BACKENDS)):
            # Always return NUMPY for np.ndarray
            if b == CUDA:
                continue
            try:
                cls = _classes[b]
            except (ImportError, ModuleNotFoundError):
                # probably no CuPy
                continue
            if isinstance(arr, cls):
                backend = b  # type: ignore
                break
        _type_cache[t] = backend
    elif backend is True:  # Just for MyPy, it can actually never become True
        raise RuntimeError()
    return backend


def get_converter(
        source_backend: Optional[ArrayBackend], target_backend: Optional[ArrayBackend],
        strict: bool = False
) -> Converter:
    '''
    Return a converter function from :code:`source_backend` to :code:`target_backend`.
    '''
    identifier = (source_backend, target_backend)
    res = _converters.get(identifier, None)
    if strict and res is None:
        raise ValueError(f"Could not find converter for {identifier}.")
    elif res is None:
        return _identity
    else:
        return res


@lru_cache(maxsize=None)
def _cheapest_pair(source_backends, target_backends) -> Tuple[ArrayBackend, ArrayBackend]:
    '''
    Actual implementation for :func:`cheapest_pair`
    '''
    # For very large sets of backends it might be more efficient to
    # sort the possible pairings by performance once and then
    # try if they match the specified backends, starting with the fastest.

    # However, for limited backend choices this is doing pretty well.
    keys = itertools.product(source_backends, target_backends)
    s = sorted(keys, key=lambda x: _cost[x])
    cheapest_key = s[0]
    return cheapest_key


def cheapest_pair(
        source_backends: Iterable[ArrayBackend], target_backends: Iterable[ArrayBackend]
) -> Tuple[ArrayBackend, ArrayBackend]:
    '''
    Find an efficient converter from source to target.

    Find the pair from the product of :code:`source_backends` and :code:`target_backends`
    with the lowest expected conversion cost.

    The cost function is currently based on hard-coded values from a simple test run.
    See :code:`scripts/benchmark.ipynb`!
    '''
    if not isinstance(source_backends, Hashable):
        source_backends = frozenset(source_backends)
    if not isinstance(target_backends, Hashable):
        target_backends = frozenset(target_backends)
    return _cheapest_pair(source_backends, target_backends)  # type: ignore


def conversion_cost(source_backend: ArrayBackend, target_backend: ArrayBackend) -> float:
    '''
    Return a floating point value that is roughly proportional
    to a typical conversion cost between the two array backends.

    The cost function is currently based on hard-coded values from a simple test run.
    See :code:`scripts/benchmark.ipynb`!
    '''
    key = (source_backend, target_backend)
    return _cost[key]


def for_backend(arr: ArrayT, backend: Optional[ArrayBackend], strict: bool = True) -> ArrayT:
    '''
    Convert :code:`arr` to the specified backend
    '''
    source_backend = get_backend(arr)
    if source_backend is None and strict:
        raise UnknownBackendError(
            f"`arr` has unknown or unsupported type `{type(arr)}`"
        )
    converter = get_converter(source_backend, backend, strict)
    # print(source_backend, backend, converter)
    return converter(arr)


def check_shape(arr: ArrayT, shape: Iterable[int]) -> None:
    '''
    Raise an exception if the shape of arr is incompatible
    with the provided shape.

    It checks if :code:`arr` only supports 2D shapes. In that case
    it expects the shape of arr to be :code: `(shape[0], prod(shape[1:]))`
    '''
    shape = tuple(shape)
    backend = get_backend(arr)
    if backend in D2_BACKENDS:
        sigprod = prod(shape[1:])
        expected = (shape[0], sigprod)
        if arr.shape != expected:
            raise ValueError(
                f"Mismatching shape {arr.shape} vs expected {expected} for {shape}, "
                f"backend {backend}."
            )
    # Also checks shape if backend is None for an unknown backend
    else:
        if arr.shape != shape:
            raise ValueError(f"Mismatching shape {arr.shape} vs {shape}, backend {backend}.")


def get_device_class(backend: Optional[ArrayBackend]) -> DeviceClass:
    '''
    Determine the device class associated with an array type

    Returns
    -------

    :code:`'cuda'` for CUDA, :code:`'cupy'` for all CuPy-based backends,
    and :code:`'cpu'` for all CPU-based backends or if the array backend is
    :code:`None` (unknown).
    '''

    if backend == CUDA:
        return 'cuda'
    elif backend in CUPY_BACKENDS:
        return 'cupy'
    elif backend in CPU_BACKENDS or backend is None:
        return 'cpu'
    else:
        raise ValueError(f"Unknown backend {backend}.")


def make_like(arr: ArrayT, target: ArrayT, strict: bool = True) -> ArrayT:
    '''
    Convert to compatible format and shape for assignment into a target array.

    The result of array operations on a sparse input array can't always be
    merged into a result array of different format directly, for example since
    arrays from the :mod:`sparse` package are not converted to NumPy arrays
    automatically and CUDA arrays may have to be transferred to CPU. This
    function takes care of that.

    This function doesn't support broadcasting, i.e. array and target shape must
    match, minus flattened dimensions for conversion from 2D to nD array
    formats.

    Parameters
    ----------

    strict
        Check shape and do strict backend conversion
    '''
    if strict:
        check_shape(arr, target.shape)
    res = for_backend(arr, get_backend(target), strict=strict).reshape(target.shape)
    return res


def benchmark_conversions(shape, dtype, density, backends=BACKENDS, repeats=10, warmup=True):
    '''
    Measure the timing for all pairwise conversions with the specified parameters.

    It can also serve as a self-test for the converter matrix.

    Parameters
    ----------

    shape
        Shape of the test array
    dtype
        Dtype of the test array
    density
        Fraction of non-zero values
    backends
        Iterable of backends to compare
    repeats
        Number of repeats to gather statistics
    warmup:
        Perform a warmup conversion before measuring. This prevents
        overheads from Numba compilation and initialization of CuPy
        from skewing the results.

    Returns
    -------

    Dictionary with (left, right) identifiers as keys and (min, max, mean) in seconds as values.
    '''
    results = {}

    def data_rvs(size):
        kind = np.dtype(dtype).kind
        if kind == 'b':
            return np.random.choice([True, False], size=size)
        elif kind == 'i':
            return np.random.randint(-128, 128, size=size, dtype=dtype)
        elif kind == 'u':
            return np.random.randint(0, 256, size=size, dtype=dtype)
        elif kind == 'f':
            return np.random.random(size).astype(dtype)
        elif kind == 'c':
            return np.random.random(size).astype(dtype) + 1j*np.random.random(size).astype(dtype)
        else:
            raise ValueError(f"Can't generate data of dtype {dtype}.")

    feedstock = sparse.random(shape=shape, density=density, data_rvs=data_rvs)
    assert feedstock.dtype == dtype

    ref = for_backend(feedstock, NUMPY)

    for left in backends:
        source = for_backend(feedstock, left)
        for right in backends:
            if warmup:
                for_backend(source, right)
            min_time = np.inf
            max_time = -np.inf
            total_time = 0
            for repeat in range(repeats):
                start = time.perf_counter_ns()
                res = for_backend(source, right)
                stop = time.perf_counter_ns()
                res_np = for_backend(res, NUMPY).reshape(ref.shape)
                assert np.allclose(ref, res_np)
                duration = stop - start
                min_time = min(min_time, duration)
                max_time = max(max_time, duration)
                total_time += duration
            res = (min_time / 1e9, max_time / 1e9, total_time / repeats / 1e9)
            results[(left, right)] = res
            print(left, right, res)

    return results
