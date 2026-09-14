"""Matrix-vector multiplication without third-party numerical libraries."""

import math
import random
from numbers import Real


def _to_numeric_list(values, name):
    """Return a validated copy of a one-dimensional numeric iterable."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of numbers, not text")

    try:
        result = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of numbers") from exc

    for index, value in enumerate(result):
        # bool is an int subclass, but treating True/False as data here usually
        # hides an input error, so reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{index}] must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"{name}[{index}] must be finite")
    return result


# Compute the dot product of two vectors using a for loop.
def dot_product(vector_a, vector_b):
    """Return the dot product of two equal-length numeric vectors."""
    a = _to_numeric_list(vector_a, "vector_a")
    b = _to_numeric_list(vector_b, "vector_b")

    # Checking lengths prevents the silent truncation that zip() would cause.
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")

    product = 0
    for index in range(len(a)):
        product += a[index] * b[index]
    return product


# Compute the matrix-vector product using dot_product for each matrix row.
def matvec_multiply(matrix, vector):
    """Return matrix @ vector, where the matrix is represented by rows."""
    if isinstance(matrix, (str, bytes)):
        raise TypeError("matrix must be an iterable of rows, not text")

    try:
        raw_rows = list(matrix)
    except TypeError as exc:
        raise TypeError("matrix must be an iterable of rows") from exc

    checked_vector = _to_numeric_list(vector, "vector")
    checked_rows = []
    expected_columns = None

    for row_index, row in enumerate(raw_rows):
        # A number or string cannot represent a matrix row.
        if isinstance(row, (Real, str, bytes)):
            raise TypeError(f"matrix row {row_index} must be an iterable of numbers")
        checked_row = _to_numeric_list(row, f"matrix[{row_index}]")

        if expected_columns is None:
            expected_columns = len(checked_row)
        elif len(checked_row) != expected_columns:
            raise ValueError("all matrix rows must have the same length")
        checked_rows.append(checked_row)

    if expected_columns is not None and expected_columns != len(checked_vector):
        raise ValueError("matrix column count must equal vector length")

    result = []
    for row in checked_rows:
        result.append(dot_product(row, checked_vector))
    return result


# Test the matrix-vector product with randomly generated 1000 x 1000 data.
def main():
    """Run a reproducible 1000 x 1000 smoke test."""
    random_generator = random.Random(500)
    size = 1000
    matrix = [
        [random_generator.uniform(-1.0, 1.0) for _ in range(size)]
        for _ in range(size)
    ]
    vector = [random_generator.uniform(-1.0, 1.0) for _ in range(size)]

    result = matvec_multiply(matrix, vector)
    assert len(result) == size
    assert all(math.isfinite(value) for value in result)
    print(f"Successfully multiplied a {size}x{size} matrix by a vector.")


if __name__ == "__main__":
    main()
