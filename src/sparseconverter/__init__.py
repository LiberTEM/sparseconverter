from functools import partial, reduce, lru_cache
import itertools
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional, Tuple, Union, Hashable
from typing_extensions import Literal

import numpy as np
import scipy.sparse as sp
import sparse


__version__ = '0.4.0.dev0'

NUMPY = 'numpy'
NUMPY_MATRIX = 'numpy.matrix'
CUDA = 'cuda'
CUPY = 'cupy'
SPARSE_COO = 'sparse.COO'
SPARSE_GCXS = 'sparse.GCXS'
SPARSE_DOK = 'sparse.DOK'

# On Python 3.7 only the matrix interface of SciPy is supported
# FIXME include scipy arrays when used with newer Python versions.
SCIPY_COO = 'scipy.sparse.coo_matrix'
SCIPY_CSR = 'scipy.sparse.csr_matrix'
SCIPY_CSC = 'scipy.sparse.csc_matrix'

CUPY_SCIPY_COO = 'cupyx.scipy.sparse.coo_matrix'
CUPY_SCIPY_CSR = 'cupyx.scipy.sparse.csr_matrix'
CUPY_SCIPY_CSC = 'cupyx.scipy.sparse.csc_matrix'

ArrayBackend = Literal[
    'numpy', 'numpy.matrix', 'cuda',
    'cupy',
    'sparse.COO', 'sparse.GCXS', 'sparse.DOK',
    'scipy.sparse.coo_matrix', 'scipy.sparse.csr_matrix', 'scipy.sparse.csc_matrix'
    'cupyx.scipy.sparse.coo_matrix', 'cupyx.scipy.sparse.csr_matrix',
    'cupyx.scipy.sparse.csc_matrix'
]

CPU_BACKENDS = frozenset((
    NUMPY, NUMPY_MATRIX, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK, SCIPY_COO, SCIPY_CSR, SCIPY_CSC
))
CUPY_BACKENDS = frozenset((CUPY, CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC))
# "on CUDA, but no CuPy" backend that receives NumPy arrays
CUDA_BACKENDS = CUPY_BACKENDS.union((CUDA, ))
BACKENDS = CPU_BACKENDS.union(CUDA_BACKENDS)
# Backends that support n-dimensional arrays as opposed to 2D-only
ND_BACKENDS = frozenset((NUMPY, CUDA, CUPY, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK))
# 2D backends
D2_BACKENDS = frozenset((
    NUMPY_MATRIX, SCIPY_COO, SCIPY_CSR, SCIPY_CSC, CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
))
# Array classes that are not ND and not 2D are thinkable, i.e. automatically assuming
# a new class falls in these two categories may be wrong.
# Keep this consistency check in case arrays are added
assert ND_BACKENDS.union(D2_BACKENDS) == BACKENDS

SPARSE_BACKENDS = frozenset((
    SPARSE_COO, SPARSE_GCXS, SPARSE_DOK,
    SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
    CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
))
DENSE_BACKENDS = BACKENDS - SPARSE_BACKENDS

DeviceClass = Literal['cpu', 'cuda', 'cupy']

if TYPE_CHECKING:
    import cupy
    import cupyx

ArrayT = Union[
    np.ndarray, np.matrix,
    sparse.SparseArray,
    sp.coo_matrix, sp.csr_matrix, sp.csc_matrix,
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
        SCIPY_COO: sp.coo_matrix,
        SCIPY_CSR: sp.csr_matrix,
        SCIPY_CSC: sp.csc_matrix
    }

    def __getitem__(self, item):
        res = self._classes.get(item, None)
        if res is None:
            return self._get_lazy(item)
        else:
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

    def _build_converters(self):
        try:
            for backend in BACKENDS:
                self._converters[(backend, backend)] = _identity
            # Both are NumPy arrays, distinguished for device selection
            for left in (NUMPY, CUDA):
                for right in (NUMPY, CUDA):
                    self._converters[(left, right)] = _identity
            self._converters[(NUMPY, NUMPY_MATRIX)] = chain(_flatsig, np.asmatrix)
            self._converters[(NUMPY_MATRIX, NUMPY)] = np.asarray

            for left in NUMPY, CUDA:
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = _classes[right].from_numpy

            formatcodes = {
                SPARSE_COO: 'coo',
                SPARSE_GCXS: 'gcxs',
                SPARSE_DOK: 'dok',
            }
            for left in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = partial(
                            _classes[left].asformat, format=formatcodes[right]
                        )

            for left in NUMPY, CUDA, SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
                for right in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = chain(_flatsig, _classes[right])
            for left in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                for right in NUMPY, CUDA:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = _classes[left].todense
            for left in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
                for right in NUMPY, CUDA:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = _classes[left].toarray
            for left in SCIPY_CSR, SCIPY_CSC:
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = _convert_csc_csr_to_pydata_sparse(
                            left, right
                        )
            for left in SCIPY_COO, :
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = _classes[right].from_scipy_sparse

            self._converters[(SPARSE_COO, SCIPY_COO)] = chain(_flatsig, sparse.COO.to_scipy_sparse)
            self._converters[(SPARSE_COO, SCIPY_CSR)] = chain(_flatsig, sparse.COO.tocsr)
            self._converters[(SPARSE_COO, SCIPY_CSC)] = chain(_flatsig, sparse.COO.tocsc)
            self._converters[(SPARSE_GCXS, SCIPY_COO)] = _GCXS_to_coo
            self._converters[(SPARSE_GCXS, SCIPY_CSR)] = _GCXS_to_csr
            self._converters[(SPARSE_GCXS, SCIPY_CSC)] = _GCXS_to_csc

            for right in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
                if (SPARSE_DOK, right) not in self._converters:
                    self._converters[(SPARSE_DOK, right)] = chain(
                        self._converters[(SPARSE_DOK, SPARSE_COO)],
                        self._converters[(SPARSE_COO, right)]
                    )

            for left in CPU_BACKENDS:
                proxy = NUMPY
                right = NUMPY_MATRIX
                if (left, right) not in self._converters:
                    c1 = self._converters[(left, proxy)]
                    c2 = self._converters[(proxy, right)]
                    self._converters[(left, right)] = chain(c1, c2)
            for right in CPU_BACKENDS:
                proxy = NUMPY
                left = NUMPY_MATRIX
                if (left, right) not in self._converters:
                    c1 = self._converters[(left, proxy)]
                    c2 = self._converters[(proxy, right)]
                    self._converters[(left, right)] = chain(c1, c2)
        except Exception:
            self._converters = None
            raise
        self._built = True

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

        self._converters[(NUMPY, CUPY)] = cupy.array
        self._converters[(CUDA, CUPY)] = cupy.array
        self._converters[(CUPY, NUMPY)] = cupy.asnumpy
        self._converters[(CUPY, CUDA)] = cupy.asnumpy

        self._converters[(CUPY_SCIPY_CSR, CUPY)] = _cupy_scipy_csr_to_CUPY

        for left in CUPY_SCIPY_COO, CUPY_SCIPY_CSC:
            if (left, CUPY) not in self._converters:
                self._converters[(left, CUPY)] = _classes[left].toarray

        for (left, right) in [
                    (CUPY_SCIPY_COO, SCIPY_COO),
                    (CUPY_SCIPY_CSR, SCIPY_CSR),
                    (CUPY_SCIPY_CSC, SCIPY_CSC)
                ]:
            if (left, right) not in self._converters:
                self._converters[(left, right)] = _classes[left].get

        # Accepted by constructor of target class
        for left in (SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
                CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC):
            for right in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
                if (left, right) not in self._converters:
                    if left in ND_BACKENDS:
                        self._converters[(left, right)] = chain(
                            _flatsig, _adjust_dtype_cupy_sparse(right), _classes[right]
                        )
                    else:
                        self._converters[(left, right)] = chain(
                            _adjust_dtype_cupy_sparse(right), _classes[right]
                        )
        # Work around https://github.com/cupy/cupy/issues/7035
        # Otherwise CUPY -> {CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC}
        # could be handled by the block above.
        self._converters[CUPY, CUPY_SCIPY_COO] = _CUPY_to_cupy_scipy_coo
        self._converters[CUPY, CUPY_SCIPY_CSR] = _CUPY_to_cupy_scipy_csr
        self._converters[CUPY, CUPY_SCIPY_CSC] = _CUPY_to_cupy_scipy_csc
        for left in NUMPY, CUDA:
            for right in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
                if (left, right) not in self._converters:
                    c1 = self._converters[(left, CUPY)]
                    c2 = self._converters[(CUPY, right)]
                    self._converters[(left, right)] = chain(c1, c2)
        self._converters[(SPARSE_GCXS, CUPY_SCIPY_COO)] = chain(
            _adjust_dtype_cupy_sparse(CUPY_SCIPY_COO), _GCXS_to_cupy_coo
        )
        self._converters[(SPARSE_GCXS, CUPY_SCIPY_CSR)] = chain(
            _adjust_dtype_cupy_sparse(CUPY_SCIPY_CSR), _GCXS_to_cupy_csr
        )
        self._converters[(SPARSE_GCXS, CUPY_SCIPY_CSC)] = chain(
            _adjust_dtype_cupy_sparse(CUPY_SCIPY_CSC), _GCXS_to_cupy_csc
        )
        self._converters[(SPARSE_GCXS, CUPY)] = _GCXS_to_cupy

        self._converters[(CUPY, SCIPY_COO)] = _CUPY_to_scipy_coo
        self._converters[(CUPY, SCIPY_CSR)] = _CUPY_to_scipy_csr
        self._converters[(CUPY, SCIPY_CSC)] = _CUPY_to_scipy_csc

        self._converters[(CUPY, SPARSE_COO)] = _CUPY_to_sparse_coo
        self._converters[(CUPY, SPARSE_GCXS)] = _CUPY_to_sparse_gcxs
        self._converters[(CUPY, SPARSE_DOK)] = _CUPY_to_sparse_dok

        self._converters[(SPARSE_COO, CUPY)] = _sparse_coo_to_CUPY
        self._converters[(SPARSE_GCXS, CUPY)] = _sparse_gcxs_to_CUPY
        self._converters[(SPARSE_DOK, CUPY)] = _sparse_dok_to_CUPY

        self._converters[(SCIPY_COO, CUPY)] = _scipy_coo_to_CUPY
        self._converters[(SCIPY_CSR, CUPY)] = _scipy_csr_to_CUPY
        self._converters[(SCIPY_CSC, CUPY)] = _scipy_csc_to_CUPY

        self._converters[(CUPY_SCIPY_CSR, SCIPY_COO)] = _cupy_scipy_csr_to_scipy_coo
        self._converters[(CUPY_SCIPY_CSR, SCIPY_CSC)] = _cupy_scipy_csr_to_scipy_csc

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
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)

        for left in BACKENDS:
            proxy = NUMPY
            right = NUMPY_MATRIX
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)
        for right in BACKENDS:
            proxy = NUMPY
            left = NUMPY_MATRIX
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)

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
    (CUDA, CUDA): 5.7087e-06,
    (CUDA, CUPY): 0.0033369292000000003,
    (CUDA, CUPY_SCIPY_COO): 0.0047700987,
    (CUDA, CUPY_SCIPY_CSC): 0.0044373838,
    (CUDA, CUPY_SCIPY_CSR): 0.0072124011,
    (CUDA, NUMPY): 5.3775e-06,
    (CUDA, NUMPY_MATRIX): 6.19358e-05,
    (CUDA, SCIPY_COO): 0.0542562901,
    (CUDA, SCIPY_CSC): 0.0579072341,
    (CUDA, SCIPY_CSR): 0.0590351847,
    (CUDA, SPARSE_COO): 0.047494878399999996,
    (CUDA, SPARSE_DOK): 0.4593789143,
    (CUDA, SPARSE_GCXS): 0.0869229908,
    (CUPY, CUDA): 0.0058407055,
    (CUPY, CUPY): 5.4837e-06,
    (CUPY, CUPY_SCIPY_COO): 0.0009950691,
    (CUPY, CUPY_SCIPY_CSC): 0.0007531936,
    (CUPY, CUPY_SCIPY_CSR): 0.0036837286,
    (CUPY, NUMPY): 0.0057900335,
    (CUPY, NUMPY_MATRIX): 0.005937204099999999,
    (CUPY, SCIPY_COO): 0.0601384683,
    (CUPY, SCIPY_CSC): 0.0645149235,
    (CUPY, SCIPY_CSR): 0.0648444757,
    (CUPY, SPARSE_COO): 0.052827728,
    (CUPY, SPARSE_DOK): 0.4642118891,
    (CUPY, SPARSE_GCXS): 0.0917574058,
    (CUPY_SCIPY_COO, CUDA): 0.0125646027,
    (CUPY_SCIPY_COO, CUPY): 0.0107726843,
    (CUPY_SCIPY_COO, CUPY_SCIPY_COO): 5.8401e-06,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSC): 0.0046786442,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSR): 0.0046695442,
    (CUPY_SCIPY_COO, NUMPY): 0.012788232699999999,
    (CUPY_SCIPY_COO, NUMPY_MATRIX): 0.012631248199999999,
    (CUPY_SCIPY_COO, SCIPY_COO): 0.001516897,
    (CUPY_SCIPY_COO, SCIPY_CSC): 0.0061664394,
    (CUPY_SCIPY_COO, SCIPY_CSR): 0.0060853895,
    (CUPY_SCIPY_COO, SPARSE_COO): 0.0098726427,
    (CUPY_SCIPY_COO, SPARSE_DOK): 1.0562659261,
    (CUPY_SCIPY_COO, SPARSE_GCXS): 0.0072367965999999995,
    (CUPY_SCIPY_CSC, CUDA): 0.0133224155,
    (CUPY_SCIPY_CSC, CUPY): 0.0061948457,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_COO): 0.0010126644,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSC): 5.9003999999999994e-06,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSR): 0.0004962211,
    (CUPY_SCIPY_CSC, NUMPY): 0.0132182667,
    (CUPY_SCIPY_CSC, NUMPY_MATRIX): 0.0132212654,
    (CUPY_SCIPY_CSC, SCIPY_COO): 0.0024322171,
    (CUPY_SCIPY_CSC, SCIPY_CSC): 0.0008341843000000001,
    (CUPY_SCIPY_CSC, SCIPY_CSR): 0.0011534851000000001,
    (CUPY_SCIPY_CSC, SPARSE_COO): 0.057683317399999996,
    (CUPY_SCIPY_CSC, SPARSE_DOK): 1.1066260407000001,
    (CUPY_SCIPY_CSC, SPARSE_GCXS): 0.0063828478,
    (CUPY_SCIPY_CSR, CUDA): 0.0107578735,
    (CUPY_SCIPY_CSR, CUPY): 0.0061654453,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_COO): 0.0010201825999999999,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC): 0.0005096146,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSR): 5.4963e-06,
    (CUPY_SCIPY_CSR, NUMPY): 0.010612198599999999,
    (CUPY_SCIPY_CSR, NUMPY_MATRIX): 0.0106934284,
    (CUPY_SCIPY_CSR, SCIPY_COO): 0.0025163781,
    (CUPY_SCIPY_CSR, SCIPY_CSC): 0.0048487582999999996,
    (CUPY_SCIPY_CSR, SCIPY_CSR): 0.0007281086999999999,
    (CUPY_SCIPY_CSR, SPARSE_COO): 0.010840902599999999,
    (CUPY_SCIPY_CSR, SPARSE_DOK): 1.0502791334,
    (CUPY_SCIPY_CSR, SPARSE_GCXS): 0.0019567613,
    (NUMPY, CUDA): 5.6909e-06,
    (NUMPY, CUPY): 0.0022502914,
    (NUMPY, CUPY_SCIPY_COO): 0.0046167395,
    (NUMPY, CUPY_SCIPY_CSC): 0.0043844903,
    (NUMPY, CUPY_SCIPY_CSR): 0.0071185418,
    (NUMPY, NUMPY): 5.3913e-06,
    (NUMPY, NUMPY_MATRIX): 6.228640000000001e-05,
    (NUMPY, SCIPY_COO): 0.0538344536,
    (NUMPY, SCIPY_CSC): 0.057908801,
    (NUMPY, SCIPY_CSR): 0.0587481384,
    (NUMPY, SPARSE_COO): 0.0475574139,
    (NUMPY, SPARSE_DOK): 0.4619955832,
    (NUMPY, SPARSE_GCXS): 0.08869084040000001,
    (NUMPY_MATRIX, CUDA): 9.0764e-06,
    (NUMPY_MATRIX, CUPY): 0.0023160705,
    (NUMPY_MATRIX, CUPY_SCIPY_COO): 0.0046872365,
    (NUMPY_MATRIX, CUPY_SCIPY_CSC): 0.0044369469000000005,
    (NUMPY_MATRIX, CUPY_SCIPY_CSR): 0.007215551099999999,
    (NUMPY_MATRIX, NUMPY): 7.9e-06,
    (NUMPY_MATRIX, NUMPY_MATRIX): 5.5805e-06,
    (NUMPY_MATRIX, SCIPY_COO): 0.054201784600000004,
    (NUMPY_MATRIX, SCIPY_CSC): 0.0582682783,
    (NUMPY_MATRIX, SCIPY_CSR): 0.0592015605,
    (NUMPY_MATRIX, SPARSE_COO): 0.0405218495,
    (NUMPY_MATRIX, SPARSE_DOK): 0.4020413502,
    (NUMPY_MATRIX, SPARSE_GCXS): 0.080696045,
    (SCIPY_COO, CUDA): 0.0068673172000000005,
    (SCIPY_COO, CUPY): 0.0104031465,
    (SCIPY_COO, CUPY_SCIPY_COO): 0.0029265753,
    (SCIPY_COO, CUPY_SCIPY_CSC): 0.0083373677,
    (SCIPY_COO, CUPY_SCIPY_CSR): 0.0075110757000000005,
    (SCIPY_COO, NUMPY): 0.007202438,
    (SCIPY_COO, NUMPY_MATRIX): 0.0071584635,
    (SCIPY_COO, SCIPY_COO): 5.3601000000000006e-06,
    (SCIPY_COO, SCIPY_CSC): 0.0038445701,
    (SCIPY_COO, SCIPY_CSR): 0.0051002343,
    (SCIPY_COO, SPARSE_COO): 0.0005566118000000001,
    (SCIPY_COO, SPARSE_DOK): 1.0414961782,
    (SCIPY_COO, SPARSE_GCXS): 0.0051015318,
    (SCIPY_CSC, CUDA): 0.0087067581,
    (SCIPY_CSC, CUPY): 0.0122098977,
    (SCIPY_CSC, CUPY_SCIPY_COO): 0.005907114,
    (SCIPY_CSC, CUPY_SCIPY_CSC): 0.0017166193999999999,
    (SCIPY_CSC, CUPY_SCIPY_CSR): 0.0068246185,
    (SCIPY_CSC, NUMPY): 0.0085353636,
    (SCIPY_CSC, NUMPY_MATRIX): 0.008607367599999999,
    (SCIPY_CSC, SCIPY_COO): 0.0036872113,
    (SCIPY_CSC, SCIPY_CSC): 6.9375e-06,
    (SCIPY_CSC, SCIPY_CSR): 0.004772579900000001,
    (SCIPY_CSC, SPARSE_COO): 0.0602223605,
    (SCIPY_CSC, SPARSE_DOK): 1.0925487128,
    (SCIPY_CSC, SPARSE_GCXS): 0.0029355244,
    (SCIPY_CSR, CUDA): 0.0103626944,
    (SCIPY_CSR, CUPY): 0.0160449881,
    (SCIPY_CSR, CUPY_SCIPY_COO): 0.0038897815,
    (SCIPY_CSR, CUPY_SCIPY_CSC): 0.0056814682,
    (SCIPY_CSR, CUPY_SCIPY_CSR): 0.0013260791,
    (SCIPY_CSR, NUMPY): 0.010360101699999999,
    (SCIPY_CSR, NUMPY_MATRIX): 0.0111408283,
    (SCIPY_CSR, SCIPY_COO): 0.0016973715,
    (SCIPY_CSR, SCIPY_CSC): 0.0040305245,
    (SCIPY_CSR, SCIPY_CSR): 5.4937e-06,
    (SCIPY_CSR, SPARSE_COO): 0.009789464199999999,
    (SCIPY_CSR, SPARSE_DOK): 1.0435496957000001,
    (SCIPY_CSR, SPARSE_GCXS): 0.0008941603000000001,
    (SPARSE_COO, CUDA): 0.0085781812,
    (SPARSE_COO, CUPY): 0.0111555192,
    (SPARSE_COO, CUPY_SCIPY_COO): 0.0213925648,
    (SPARSE_COO, CUPY_SCIPY_CSC): 0.0268739308,
    (SPARSE_COO, CUPY_SCIPY_CSR): 0.0238861197,
    (SPARSE_COO, NUMPY): 0.008692947199999999,
    (SPARSE_COO, NUMPY_MATRIX): 0.008783360300000001,
    (SPARSE_COO, SCIPY_COO): 0.0171435471,
    (SPARSE_COO, SCIPY_CSC): 0.021895475100000002,
    (SPARSE_COO, SCIPY_CSR): 0.0179420075,
    (SPARSE_COO, SPARSE_COO): 5.3912e-06,
    (SPARSE_COO, SPARSE_DOK): 1.1438241113,
    (SPARSE_COO, SPARSE_GCXS): 0.0418598023,
    (SPARSE_DOK, CUDA): 0.1219599234,
    (SPARSE_DOK, CUPY): 0.123253974,
    (SPARSE_DOK, CUPY_SCIPY_COO): 0.5575438017000001,
    (SPARSE_DOK, CUPY_SCIPY_CSC): 0.5585472227999999,
    (SPARSE_DOK, CUPY_SCIPY_CSR): 0.5613383429,
    (SPARSE_DOK, NUMPY): 0.1239053871,
    (SPARSE_DOK, NUMPY_MATRIX): 0.12213457929999999,
    (SPARSE_DOK, SCIPY_COO): 0.56037954,
    (SPARSE_DOK, SCIPY_CSC): 0.5572885227000001,
    (SPARSE_DOK, SCIPY_CSR): 0.5473552376,
    (SPARSE_DOK, SPARSE_COO): 0.5275931475,
    (SPARSE_DOK, SPARSE_DOK): 5.5684e-06,
    (SPARSE_DOK, SPARSE_GCXS): 0.5786749655,
    (SPARSE_GCXS, CUDA): 0.0415628543,
    (SPARSE_GCXS, CUPY): 0.0440102195,
    (SPARSE_GCXS, CUPY_SCIPY_COO): 0.2519524705,
    (SPARSE_GCXS, CUPY_SCIPY_CSC): 0.2550338915,
    (SPARSE_GCXS, CUPY_SCIPY_CSR): 0.250026516,
    (SPARSE_GCXS, NUMPY): 0.0417246977,
    (SPARSE_GCXS, NUMPY_MATRIX): 0.0416986575,
    (SPARSE_GCXS, SCIPY_COO): 0.15432492269999998,
    (SPARSE_GCXS, SCIPY_CSC): 0.15875735259999998,
    (SPARSE_GCXS, SCIPY_CSR): 0.1543013657,
    (SPARSE_GCXS, SPARSE_COO): 0.028974817,
    (SPARSE_GCXS, SPARSE_DOK): 1.1691816655,
    (SPARSE_GCXS, SPARSE_GCXS): 5.6678999999999996e-06,
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
        for b in (NUMPY_MATRIX, ) + tuple(BACKENDS):
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
