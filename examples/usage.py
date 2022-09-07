import numpy as np
from sparseconverter import (
    NUMPY, SCIPY_COO, SCIPY_CSC, SCIPY_CSR, SPARSE_COO, SPARSE_DOK, SPARSE_GCXS,
    cheapest_pair, check_shape, for_backend, get_backend, get_converter, get_device_class, make_like
)


def main():
    a1 = np.array([
        (1, 0, 3),
        (0, 0, 6)
    ])

    # array conversion
    a2 = for_backend(a1, SPARSE_GCXS)

    # format determination
    print("a2 is", get_backend(a2))

    # Get transformation function
    # Can potentially save the lookup step in an inner loop
    transform_function = get_converter(SPARSE_GCXS, SCIPY_CSR)
    a3 = transform_function(a2)
    print("a3 is", get_backend(a3))

    # Which format is probably fastest to create from NumPy?
    _, right = cheapest_pair(
        source_backends=(NUMPY, ),
        target_backends=(SPARSE_GCXS, SPARSE_COO, SPARSE_DOK, SCIPY_COO, SCIPY_CSC, SCIPY_CSR)
    )
    print(f"Converting from NumPy to {right} is probably cheapest.")

    # Does our output have a shape that is compatible with a target shape?
    check_shape(a3, a1.shape)

    # CPU or CuPy?
    print("Device class of a3 is", get_device_class(get_backend(a3)))

    # Adjust shape and format to match target array
    a4 = make_like(a3, a1)

    print("a4 is", get_backend(a4))


if __name__ == '__main__':
    main()
