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
        self._converters = {}
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
                        SCIPY_COO, SCIPY_CSR, SCIPY_CSC
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
            for left in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
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

        has_csr_complex128_bug = detect_csr_complex128_bug()
        has_cupy_coo_construction = detect_coo_works()

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
                return sparse.GCXS(intermediate.get()).reshape(arr.shape)
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

        self._converters[(NUMPY, CUPY)] = cupy.array
        self._converters[(CUDA, CUPY)] = cupy.array
        self._converters[(CUPY, NUMPY)] = cupy.asnumpy
        self._converters[(CUPY, CUDA)] = cupy.asnumpy

        for left in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
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
    (CUDA, CUDA): 9.52e-06,
    (CUDA, CUPY): 0.0034923,
    (CUDA, CUPY_SCIPY_COO): 0.02723361,
    (CUDA, CUPY_SCIPY_CSC): 0.04614623,
    (CUDA, CUPY_SCIPY_CSR): 0.03003148,
    (CUDA, NUMPY): 1.158e-05,
    (CUDA, NUMPY_MATRIX): 7.85e-05,
    (CUDA, SCIPY_COO): 0.07095157,
    (CUDA, SCIPY_CSC): 0.08312156,
    (CUDA, SCIPY_CSR): 0.08185261,
    (CUDA, SPARSE_COO): 0.05721102,
    (CUDA, SPARSE_DOK): 0.47822719,
    (CUDA, SPARSE_GCXS): 0.0926257,
    (CUPY, CUDA): 0.04123289,
    (CUPY, CUPY): 9.94e-06,
    (CUPY, CUPY_SCIPY_COO): 0.02365191,
    (CUPY, CUPY_SCIPY_CSC): 0.01339789,
    (CUPY, CUPY_SCIPY_CSR): 0.02236629,
    (CUPY, NUMPY): 0.04115815,
    (CUPY, NUMPY_MATRIX): 0.04145609,
    (CUPY, SCIPY_COO): 0.12031532,
    (CUPY, SCIPY_CSC): 0.1171756,
    (CUPY, SCIPY_CSR): 0.11456013,
    (CUPY, SPARSE_COO): 0.1018334,
    (CUPY, SPARSE_DOK): 0.52614732,
    (CUPY, SPARSE_GCXS): 0.14133173,
    (CUPY_SCIPY_COO, CUDA): 0.01391734,
    (CUPY_SCIPY_COO, CUPY): 0.08673978,
    (CUPY_SCIPY_COO, CUPY_SCIPY_COO): 9.15e-06,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSC): 0.03534527,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSR): 0.03556132,
    (CUPY_SCIPY_COO, NUMPY): 0.01307202,
    (CUPY_SCIPY_COO, NUMPY_MATRIX): 0.02308691,
    (CUPY_SCIPY_COO, SCIPY_COO): 0.01752424,
    (CUPY_SCIPY_COO, SCIPY_CSC): 0.04930567,
    (CUPY_SCIPY_COO, SCIPY_CSR): 0.04840724,
    (CUPY_SCIPY_COO, SPARSE_COO): 0.02267181,
    (CUPY_SCIPY_COO, SPARSE_DOK): 0.81842092,
    (CUPY_SCIPY_COO, SPARSE_GCXS): 0.04826163,
    (CUPY_SCIPY_CSC, CUDA): 0.0168752,
    (CUPY_SCIPY_CSC, CUPY): 0.04769173,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_COO): 0.02170808,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSC): 7.97e-06,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSR): 0.02813724,
    (CUPY_SCIPY_CSC, NUMPY): 0.01218606,
    (CUPY_SCIPY_CSC, NUMPY_MATRIX): 0.01833548,
    (CUPY_SCIPY_CSC, SCIPY_COO): 0.01265958,
    (CUPY_SCIPY_CSC, SCIPY_CSC): 0.00424232,
    (CUPY_SCIPY_CSC, SCIPY_CSR): 0.02838422,
    (CUPY_SCIPY_CSC, SPARSE_COO): 0.12949171,
    (CUPY_SCIPY_CSC, SPARSE_DOK): 0.91622471,
    (CUPY_SCIPY_CSC, SPARSE_GCXS): 0.01107186,
    (CUPY_SCIPY_CSR, CUDA): 0.01544975,
    (CUPY_SCIPY_CSR, CUPY): 0.04449919,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_COO): 0.01864054,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC): 0.03305878,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSR): 9.44e-06,
    (CUPY_SCIPY_CSR, NUMPY): 0.01093995,
    (CUPY_SCIPY_CSR, NUMPY_MATRIX): 0.01565349,
    (CUPY_SCIPY_CSR, SCIPY_COO): 0.0187434,
    (CUPY_SCIPY_CSR, SCIPY_CSC): 0.01831002,
    (CUPY_SCIPY_CSR, SCIPY_CSR): 0.00920348,
    (CUPY_SCIPY_CSR, SPARSE_COO): 0.02926526,
    (CUPY_SCIPY_CSR, SPARSE_DOK): 0.81829223,
    (CUPY_SCIPY_CSR, SPARSE_GCXS): 0.00943145,
    (NUMPY, CUDA): 8.96e-06,
    (NUMPY, CUPY): 0.00304166,
    (NUMPY, CUPY_SCIPY_COO): 0.03111887,
    (NUMPY, CUPY_SCIPY_CSC): 0.03194446,
    (NUMPY, CUPY_SCIPY_CSR): 0.02707654,
    (NUMPY, NUMPY): 8.62e-06,
    (NUMPY, NUMPY_MATRIX): 8.378e-05,
    (NUMPY, SCIPY_COO): 0.07762542,
    (NUMPY, SCIPY_CSC): 0.08492988,
    (NUMPY, SCIPY_CSR): 0.07794126,
    (NUMPY, SPARSE_COO): 0.05836581,
    (NUMPY, SPARSE_DOK): 0.48218718,
    (NUMPY, SPARSE_GCXS): 0.09733027,
    (NUMPY_MATRIX, CUDA): 1.267e-05,
    (NUMPY_MATRIX, CUPY): 0.00304186,
    (NUMPY_MATRIX, CUPY_SCIPY_COO): 0.03103336,
    (NUMPY_MATRIX, CUPY_SCIPY_CSC): 0.03340842,
    (NUMPY_MATRIX, CUPY_SCIPY_CSR): 0.02498352,
    (NUMPY_MATRIX, NUMPY): 1.266e-05,
    (NUMPY_MATRIX, NUMPY_MATRIX): 9.09e-06,
    (NUMPY_MATRIX, SCIPY_COO): 0.07374739,
    (NUMPY_MATRIX, SCIPY_CSC): 0.08479951,
    (NUMPY_MATRIX, SCIPY_CSR): 0.07748237,
    (NUMPY_MATRIX, SPARSE_COO): 0.05546078,
    (NUMPY_MATRIX, SPARSE_DOK): 0.42378837,
    (NUMPY_MATRIX, SPARSE_GCXS): 0.08626198,
    (SCIPY_COO, CUDA): 0.00558513,
    (SCIPY_COO, CUPY): 0.01156896,
    (SCIPY_COO, CUPY_SCIPY_COO): 0.02330746,
    (SCIPY_COO, CUPY_SCIPY_CSC): 0.01858102,
    (SCIPY_COO, CUPY_SCIPY_CSR): 0.01082556,
    (SCIPY_COO, NUMPY): 0.00653755,
    (SCIPY_COO, NUMPY_MATRIX): 0.00598033,
    (SCIPY_COO, SCIPY_COO): 8.16e-06,
    (SCIPY_COO, SCIPY_CSC): 0.01044739,
    (SCIPY_COO, SCIPY_CSR): 0.00378696,
    (SCIPY_COO, SPARSE_COO): 0.00141525,
    (SCIPY_COO, SPARSE_DOK): 0.80436851,
    (SCIPY_COO, SPARSE_GCXS): 0.00408728,
    (SCIPY_CSC, CUDA): 0.00567477,
    (SCIPY_CSC, CUPY): 0.01150378,
    (SCIPY_CSC, CUPY_SCIPY_COO): 0.02839993,
    (SCIPY_CSC, CUPY_SCIPY_CSC): 0.00359595,
    (SCIPY_CSC, CUPY_SCIPY_CSR): 0.00917898,
    (SCIPY_CSC, NUMPY): 0.00634705,
    (SCIPY_CSC, NUMPY_MATRIX): 0.00548948,
    (SCIPY_CSC, SCIPY_COO): 0.0104994,
    (SCIPY_CSC, SCIPY_CSC): 8.02e-06,
    (SCIPY_CSC, SCIPY_CSR): 0.00648227,
    (SCIPY_CSC, SPARSE_COO): 0.12335432,
    (SCIPY_CSC, SPARSE_DOK): 0.91889736,
    (SCIPY_CSC, SPARSE_GCXS): 7.436e-05,
    (SCIPY_CSR, CUDA): 0.00568018,
    (SCIPY_CSR, CUPY): 0.0096229,
    (SCIPY_CSR, CUPY_SCIPY_COO): 0.02727776,
    (SCIPY_CSR, CUPY_SCIPY_CSC): 0.01258067,
    (SCIPY_CSR, CUPY_SCIPY_CSR): 0.00259827,
    (SCIPY_CSR, NUMPY): 0.00597631,
    (SCIPY_CSR, NUMPY_MATRIX): 0.0056923,
    (SCIPY_CSR, SCIPY_COO): 0.00909677,
    (SCIPY_CSR, SCIPY_CSC): 0.01083296,
    (SCIPY_CSR, SCIPY_CSR): 8.53e-06,
    (SCIPY_CSR, SPARSE_COO): 0.01889156,
    (SCIPY_CSR, SPARSE_DOK): 0.81050336,
    (SCIPY_CSR, SPARSE_GCXS): 6.774e-05,
    (SPARSE_COO, CUDA): 0.01559104,
    (SPARSE_COO, CUPY): 0.02050667,
    (SPARSE_COO, CUPY_SCIPY_COO): 0.04699705,
    (SPARSE_COO, CUPY_SCIPY_CSC): 0.03912978,
    (SPARSE_COO, CUPY_SCIPY_CSR): 0.03297643,
    (SPARSE_COO, NUMPY): 0.01665228,
    (SPARSE_COO, NUMPY_MATRIX): 0.01669986,
    (SPARSE_COO, SCIPY_COO): 0.02236511,
    (SPARSE_COO, SCIPY_CSC): 0.03321482,
    (SPARSE_COO, SCIPY_CSR): 0.02309729,
    (SPARSE_COO, SPARSE_COO): 9.63e-06,
    (SPARSE_COO, SPARSE_DOK): 0.86456496,
    (SPARSE_COO, SPARSE_GCXS): 0.03934677,
    (SPARSE_DOK, CUDA): 0.11838129,
    (SPARSE_DOK, CUPY): 0.12566395,
    (SPARSE_DOK, CUPY_SCIPY_COO): 0.46361514,
    (SPARSE_DOK, CUPY_SCIPY_CSC): 0.48294454,
    (SPARSE_DOK, CUPY_SCIPY_CSR): 0.48472009,
    (SPARSE_DOK, NUMPY): 0.12760996,
    (SPARSE_DOK, NUMPY_MATRIX): 0.12691516,
    (SPARSE_DOK, SCIPY_COO): 0.47121842,
    (SPARSE_DOK, SCIPY_CSC): 0.45739357,
    (SPARSE_DOK, SCIPY_CSR): 0.45030841,
    (SPARSE_DOK, SPARSE_COO): 0.43964837,
    (SPARSE_DOK, SPARSE_DOK): 8.7e-06,
    (SPARSE_DOK, SPARSE_GCXS): 0.4815116,
    (SPARSE_GCXS, CUDA): 0.05721252,
    (SPARSE_GCXS, CUPY): 0.05968288,
    (SPARSE_GCXS, CUPY_SCIPY_COO): 0.54629951,
    (SPARSE_GCXS, CUPY_SCIPY_CSC): 0.51329979,
    (SPARSE_GCXS, CUPY_SCIPY_CSR): 0.51411286,
    (SPARSE_GCXS, NUMPY): 0.05773925,
    (SPARSE_GCXS, NUMPY_MATRIX): 0.05410605,
    (SPARSE_GCXS, SCIPY_COO): 0.44208745,
    (SPARSE_GCXS, SCIPY_CSC): 0.43770924,
    (SPARSE_GCXS, SCIPY_CSR): 0.43229283,
    (SPARSE_GCXS, SPARSE_COO): 0.03560626,
    (SPARSE_GCXS, SPARSE_DOK): 0.88085017,
    (SPARSE_GCXS, SPARSE_GCXS): 2.003e-05,
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
