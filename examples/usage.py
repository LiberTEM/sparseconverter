import numpy as np

import sparseconverter as spc


def main():
    a1 = np.array([
        (1, 0, 3),
        (0, 0, 6)
    ])

    # array conversion
    a2 = spc.for_backend(a1, spc.SPARSE_GCXS)

    # format determination
    print("a2 is", spc.get_backend(a2))

    # Get transformation function
    # Can potentially save the lookup step in an inner loop
    transform_function = spc.get_converter(spc.SPARSE_GCXS, spc.SCIPY_CSR)
    a3 = transform_function(a2)
    print("a3 is", spc.get_backend(a3))

    # Which format is probably fastest to create from NumPy?
    left, right = spc.cheapest_pair(
        source_backends=(spc.NUMPY, ),
        target_backends=(
            spc.SPARSE_GCXS, spc.SPARSE_COO, spc.SPARSE_DOK,
            spc.SCIPY_COO, spc.SCIPY_CSC, spc.SCIPY_CSR
        )
    )
    cost = spc.conversion_cost(left, right)
    print(f"Converting from NumPy to {right} is probably cheapest.")
    print(f"The cost factor of converting NumPy to {right} is {cost}.")

    # Does our output have a shape that is compatible with a target shape?
    spc.check_shape(a3, a1.shape)

    # CPU or CuPy?
    print("Device class of a3 is", spc.get_device_class(spc.get_backend(a3)))

    # Adjust shape and format to match target array
    a4 = spc.make_like(a3, a1)

    print("a4 is", spc.get_backend(a4))


if __name__ == '__main__':
    main()
