from functools import partial, reduce, lru_cache
import itertools
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional, Tuple, Union, Hashable
from typing_extensions import Literal

import numpy as np
import scipy.sparse as sp
import sparse


__version__ = '0.3.4'

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
            # Support direct construction from each other
            for left in (
                        NUMPY, CUDA, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK,
                    ):
                for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                    if (left, right) not in self._converters:
                        self._converters[(left, right)] = _classes[right]
            # Overwrite from before
            self._converters[(SPARSE_DOK, SPARSE_GCXS)] = partial(
                sparse.DOK.asformat, format='gcxs'
            )
            self._converters[(SPARSE_GCXS, SPARSE_DOK)] = partial(
                sparse.GCXS.asformat, format='dok'
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
    (CUDA, CUDA): 1.397e-05,
    (CUDA, CUPY): 0.00290094,
    (CUDA, CUPY_SCIPY_COO): 0.03676262,
    (CUDA, CUPY_SCIPY_CSC): 0.02478382,
    (CUDA, CUPY_SCIPY_CSR): 0.03029807,
    (CUDA, NUMPY): 1.101e-05,
    (CUDA, NUMPY_MATRIX): 0.00014465,
    (CUDA, SCIPY_COO): 0.07126909,
    (CUDA, SCIPY_CSC): 0.07710105,
    (CUDA, SCIPY_CSR): 0.07387733,
    (CUDA, SPARSE_COO): 0.06164377,
    (CUDA, SPARSE_DOK): 0.41626783,
    (CUDA, SPARSE_GCXS): 0.08713908,
    (CUPY, CUDA): 0.04128729,
    (CUPY, CUPY): 1.315e-05,
    (CUPY, CUPY_SCIPY_COO): 0.03479637,
    (CUPY, CUPY_SCIPY_CSC): 0.02891527,
    (CUPY, CUPY_SCIPY_CSR): 0.02077345,
    (CUPY, NUMPY): 0.04131443,
    (CUPY, NUMPY_MATRIX): 0.04139519,
    (CUPY, SCIPY_COO): 0.11484382,
    (CUPY, SCIPY_CSC): 0.12240084,
    (CUPY, SCIPY_CSR): 0.11560138,
    (CUPY, SPARSE_COO): 0.08293617,
    (CUPY, SPARSE_DOK): 0.45851607,
    (CUPY, SPARSE_GCXS): 0.13566548,
    (CUPY_SCIPY_COO, CUDA): 0.01279124,
    (CUPY_SCIPY_COO, CUPY): 0.08431704,
    (CUPY_SCIPY_COO, CUPY_SCIPY_COO): 9.8e-06,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSC): 0.03095062,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSR): 0.03350027,
    (CUPY_SCIPY_COO, NUMPY): 0.01959753,
    (CUPY_SCIPY_COO, NUMPY_MATRIX): 0.01955723,
    (CUPY_SCIPY_COO, SCIPY_COO): 0.0140366,
    (CUPY_SCIPY_COO, SCIPY_CSC): 0.05566742,
    (CUPY_SCIPY_COO, SCIPY_CSR): 0.04540595,
    (CUPY_SCIPY_COO, SPARSE_COO): 0.02240446,
    (CUPY_SCIPY_COO, SPARSE_DOK): 0.81763242,
    (CUPY_SCIPY_COO, SPARSE_GCXS): 0.0503551,
    (CUPY_SCIPY_CSC, CUDA): 0.01170401,
    (CUPY_SCIPY_CSC, CUPY): 0.04833271,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_COO): 0.0154935,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSC): 1.079e-05,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSR): 0.01317206,
    (CUPY_SCIPY_CSC, NUMPY): 0.01652527,
    (CUPY_SCIPY_CSC, NUMPY_MATRIX): 0.01623056,
    (CUPY_SCIPY_CSC, SCIPY_COO): 0.03094247,
    (CUPY_SCIPY_CSC, SCIPY_CSC): 0.00378959,
    (CUPY_SCIPY_CSC, SCIPY_CSR): 0.04601307,
    (CUPY_SCIPY_CSC, SPARSE_COO): 0.09272055,
    (CUPY_SCIPY_CSC, SPARSE_DOK): 0.87353325,
    (CUPY_SCIPY_CSC, SPARSE_GCXS): 0.01177203,
    (CUPY_SCIPY_CSR, CUDA): 0.01429861,
    (CUPY_SCIPY_CSR, CUPY): 0.04460615,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_COO): 0.01443459,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC): 0.002124,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSR): 1.167e-05,
    (CUPY_SCIPY_CSR, NUMPY): 0.01489683,
    (CUPY_SCIPY_CSR, NUMPY_MATRIX): 0.01469383,
    (CUPY_SCIPY_CSR, SCIPY_COO): 0.01336538,
    (CUPY_SCIPY_CSR, SCIPY_CSC): 0.0139769,
    (CUPY_SCIPY_CSR, SCIPY_CSR): 0.00912916,
    (CUPY_SCIPY_CSR, SPARSE_COO): 0.02864303,
    (CUPY_SCIPY_CSR, SPARSE_DOK): 0.80137306,
    (CUPY_SCIPY_CSR, SPARSE_GCXS): 0.00764849,
    (NUMPY, CUDA): 1.228e-05,
    (NUMPY, CUPY): 0.00275125,
    (NUMPY, CUPY_SCIPY_COO): 0.03072501,
    (NUMPY, CUPY_SCIPY_CSC): 0.03698203,
    (NUMPY, CUPY_SCIPY_CSR): 0.02565548,
    (NUMPY, NUMPY): 2.177e-05,
    (NUMPY, NUMPY_MATRIX): 0.00011643,
    (NUMPY, SCIPY_COO): 0.071367,
    (NUMPY, SCIPY_CSC): 0.08173664,
    (NUMPY, SCIPY_CSR): 0.07519809,
    (NUMPY, SPARSE_COO): 0.06172041,
    (NUMPY, SPARSE_DOK): 0.41673362,
    (NUMPY, SPARSE_GCXS): 0.08833201,
    (NUMPY_MATRIX, CUDA): 1.812e-05,
    (NUMPY_MATRIX, CUPY): 0.00284773,
    (NUMPY_MATRIX, CUPY_SCIPY_COO): 0.02692905,
    (NUMPY_MATRIX, CUPY_SCIPY_CSC): 0.0248778,
    (NUMPY_MATRIX, CUPY_SCIPY_CSR): 0.02930477,
    (NUMPY_MATRIX, NUMPY): 3.088e-05,
    (NUMPY_MATRIX, NUMPY_MATRIX): 1.357e-05,
    (NUMPY_MATRIX, SCIPY_COO): 0.07221178,
    (NUMPY_MATRIX, SCIPY_CSC): 0.08215391,
    (NUMPY_MATRIX, SCIPY_CSR): 0.07470251,
    (NUMPY_MATRIX, SPARSE_COO): 0.05499004,
    (NUMPY_MATRIX, SPARSE_DOK): 0.36448452,
    (NUMPY_MATRIX, SPARSE_GCXS): 0.07961737,
    (SCIPY_COO, CUDA): 0.00554266,
    (SCIPY_COO, CUPY): 0.01049599,
    (SCIPY_COO, CUPY_SCIPY_COO): 0.02096826,
    (SCIPY_COO, CUPY_SCIPY_CSC): 0.01695301,
    (SCIPY_COO, CUPY_SCIPY_CSR): 0.00833004,
    (SCIPY_COO, NUMPY): 0.0057289,
    (SCIPY_COO, NUMPY_MATRIX): 0.00514385,
    (SCIPY_COO, SCIPY_COO): 1.483e-05,
    (SCIPY_COO, SCIPY_CSC): 0.00908695,
    (SCIPY_COO, SCIPY_CSR): 0.00409992,
    (SCIPY_COO, SPARSE_COO): 0.0015528,
    (SCIPY_COO, SPARSE_DOK): 0.76291548,
    (SCIPY_COO, SPARSE_GCXS): 0.00415204,
    (SCIPY_CSC, CUDA): 0.00534249,
    (SCIPY_CSC, CUPY): 0.01051322,
    (SCIPY_CSC, CUPY_SCIPY_COO): 0.02545826,
    (SCIPY_CSC, CUPY_SCIPY_CSC): 0.00291492,
    (SCIPY_CSC, CUPY_SCIPY_CSR): 0.00827505,
    (SCIPY_CSC, NUMPY): 0.00567323,
    (SCIPY_CSC, NUMPY_MATRIX): 0.00572489,
    (SCIPY_CSC, SCIPY_COO): 0.00559732,
    (SCIPY_CSC, SCIPY_CSC): 1.329e-05,
    (SCIPY_CSC, SCIPY_CSR): 0.00556892,
    (SCIPY_CSC, SPARSE_COO): 0.07105924,
    (SCIPY_CSC, SPARSE_DOK): 0.82178102,
    (SCIPY_CSC, SPARSE_GCXS): 0.00404301,
    (SCIPY_CSR, CUDA): 0.00493635,
    (SCIPY_CSR, CUPY): 0.01012645,
    (SCIPY_CSR, CUPY_SCIPY_COO): 0.02334584,
    (SCIPY_CSR, CUPY_SCIPY_CSC): 0.01313419,
    (SCIPY_CSR, CUPY_SCIPY_CSR): 0.00245257,
    (SCIPY_CSR, NUMPY): 0.00552472,
    (SCIPY_CSR, NUMPY_MATRIX): 0.00569328,
    (SCIPY_CSR, SCIPY_COO): 0.00398569,
    (SCIPY_CSR, SCIPY_CSC): 0.00942009,
    (SCIPY_CSR, SCIPY_CSR): 1.459e-05,
    (SCIPY_CSR, SPARSE_COO): 0.01763324,
    (SCIPY_CSR, SPARSE_DOK): 0.83002728,
    (SCIPY_CSR, SPARSE_GCXS): 0.00215682,
    (SPARSE_COO, CUDA): 0.01227037,
    (SPARSE_COO, CUPY): 0.01397232,
    (SPARSE_COO, CUPY_SCIPY_COO): 0.03187509,
    (SPARSE_COO, CUPY_SCIPY_CSC): 0.03986542,
    (SPARSE_COO, CUPY_SCIPY_CSR): 0.02893439,
    (SPARSE_COO, NUMPY): 0.0098226,
    (SPARSE_COO, NUMPY_MATRIX): 0.01040078,
    (SPARSE_COO, SCIPY_COO): 0.02069974,
    (SPARSE_COO, SCIPY_CSC): 0.03106024,
    (SPARSE_COO, SCIPY_CSR): 0.02506089,
    (SPARSE_COO, SPARSE_COO): 1.328e-05,
    (SPARSE_COO, SPARSE_DOK): 0.83920339,
    (SPARSE_COO, SPARSE_GCXS): 0.03594461,
    (SPARSE_DOK, CUDA): 0.11058626,
    (SPARSE_DOK, CUPY): 0.10950506,
    (SPARSE_DOK, CUPY_SCIPY_COO): 0.44371138,
    (SPARSE_DOK, CUPY_SCIPY_CSC): 0.4265934,
    (SPARSE_DOK, CUPY_SCIPY_CSR): 0.43226047,
    (SPARSE_DOK, NUMPY): 0.11354399,
    (SPARSE_DOK, NUMPY_MATRIX): 0.10729896,
    (SPARSE_DOK, SCIPY_COO): 0.42308864,
    (SPARSE_DOK, SCIPY_CSC): 0.45172044,
    (SPARSE_DOK, SCIPY_CSR): 0.42915226,
    (SPARSE_DOK, SPARSE_COO): 0.41190378,
    (SPARSE_DOK, SPARSE_DOK): 1.187e-05,
    (SPARSE_DOK, SPARSE_GCXS): 0.4325561,
    (SPARSE_GCXS, CUDA): 0.05120677,
    (SPARSE_GCXS, CUPY): 0.05303882,
    (SPARSE_GCXS, CUPY_SCIPY_COO): 0.47022746,
    (SPARSE_GCXS, CUPY_SCIPY_CSC): 0.45507096,
    (SPARSE_GCXS, CUPY_SCIPY_CSR): 0.44203396,
    (SPARSE_GCXS, NUMPY): 0.04957471,
    (SPARSE_GCXS, NUMPY_MATRIX): 0.04963172,
    (SPARSE_GCXS, SCIPY_COO): 0.36567051,
    (SPARSE_GCXS, SCIPY_CSC): 0.34491236,
    (SPARSE_GCXS, SCIPY_CSR): 0.32723106,
    (SPARSE_GCXS, SPARSE_COO): 0.03921757,
    (SPARSE_GCXS, SPARSE_DOK): 0.87211339,
    (SPARSE_GCXS, SPARSE_GCXS): 1.298e-05,
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
    converter = get_converter(get_backend(arr), backend, strict)
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
