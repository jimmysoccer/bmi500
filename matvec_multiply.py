"""Matrix-vector multiplication implemented from scratch in pure Python.

This module implements two functions required by the assignment:

* ``dot_product``      -- dot product of two vectors, computed with a ``for`` loop.
* ``matvec_multiply``  -- matrix-vector product ``A @ x`` built on top of ``dot_product``.
* ``main``             -- self-test on randomly generated data (default 1000x1000).

Everything is standard-library only (``random`` + ``time`` + ``numbers``), so no
NumPy is needed.  Inputs are validated before any arithmetic happens, so a bad
input raises a clear ``TypeError`` / ``ValueError`` instead of silently
producing a wrong number (see the failure-mode notes in ``AI_code_gen.txt``).
"""

from __future__ import annotations

import math
import random
import time
from numbers import Number
from typing import Iterable, List, Sequence

__all__ = ["dot_product", "matvec_multiply", "main"]

# Sequences we must reject: they are iterable, but iterating them character by
# character (or byte by byte) is never what a caller means by "a vector".
_STRING_LIKE = (str, bytes, bytearray)


def _as_list(name: str, values: Sequence[Number]) -> List[Number]:
    """Validate that ``values`` is a non-string iterable and return it as a list.

    Converting to a list up front does three things:
      1. it rejects genuinely non-iterable objects (ints, None, objects...);
      2. it materialises generators so we can take ``len()`` and iterate twice;
      3. it gives every downstream loop a stable snapshot of the data.
    """
    if isinstance(values, _STRING_LIKE):
        raise TypeError(
            f"{name} must be a sequence of numbers, got {type(values).__name__} "
            f"(strings/bytes are iterable but are not numeric vectors)"
        )
    if isinstance(values, Number):
        # A bare scalar is iterable-safe to reject early with a clear message,
        # otherwise users get a confusing "int object is not iterable" later.
        raise TypeError(
            f"{name} must be a sequence of numbers, got a bare scalar "
            f"({type(values).__name__})"
        )
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable sequence of numbers, "
            f"got {type(values).__name__}"
        ) from exc


def _check_numbers(
    name: str, values: Sequence[Number], allow_nonfinite: bool
) -> None:
    """Validate that every entry of ``values`` is a usable number.

    Guards three failure modes:
      * non-numeric entries (``"3"`` would otherwise be silently concatenated);
      * booleans, which are ``int`` subclasses -- ``True * True`` == ``1`` gives
        answers that look plausible but are almost always a bug upstream;
      * NaN / infinity, which poison every later result without raising.
    """
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Number):
            raise TypeError(
                f"{name}[{index}] must be a real/complex number, "
                f"got {type(value).__name__} ({value!r})"
            )
        if not allow_nonfinite and isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"{name}[{index}] is not finite ({value!r}); pass "
                f"allow_nonfinite=True to compute with NaN/inf anyway"
            )


def dot_product(
    vector_a: Sequence[Number],
    vector_b: Sequence[Number],
    allow_nonfinite: bool = False,
) -> Number:
    """Return the dot product ``sum(a[i] * b[i])`` of two equal-length vectors.

    Computed with an explicit ``for`` loop as required -- no ``sum()`` builtin,
    no list comprehension, no NumPy.

    Parameters
    ----------
    vector_a, vector_b:
        Sequences (list, tuple, generator, range, ...) of numbers.  Both must
        have the same length.
    allow_nonfinite:
        When ``False`` (the default) NaN/inf entries raise ``ValueError``.

    Returns
    -------
    Number
        ``0`` for two empty vectors (the empty sum), otherwise the scalar
        product.  The concrete type follows Python arithmetic rules:
        int-only inputs give an ``int``, any float gives a ``float``.

    Raises
    ------
    TypeError
        If either argument is not a numeric sequence (including bare scalars
        and strings), or contains a non-numeric / boolean entry.
    ValueError
        If the two vectors have different lengths, or a non-finite entry is
        present while ``allow_nonfinite`` is ``False``.
    """
    a = _as_list("vector_a", vector_a)
    b = _as_list("vector_b", vector_b)

    # Dimension check first: a zip() loop would otherwise truncate silently at
    # the shorter vector and return a believable but wrong number.
    if len(a) != len(b):
        raise ValueError(
            f"dimension mismatch: len(vector_a)={len(a)} != len(vector_b)={len(b)}"
        )

    _check_numbers("vector_a", a, allow_nonfinite)
    _check_numbers("vector_b", b, allow_nonfinite)

    # --- the required for loop -------------------------------------------
    # Start at integer 0 so int * int stays exact, and complex numbers work.
    total: Number = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def matvec_multiply(
    matrix: Sequence[Sequence[Number]],
    vector: Sequence[Number],
    allow_nonfinite: bool = False,
) -> List[Number]:
    """Return ``matrix @ vector`` for a matrix stored as a list of rows.

    Entry ``i`` of the result is ``dot_product(matrix[i], vector)``: the dot
    product of the ``i``-th row with the vector.

    Parameters
    ----------
    matrix:
        Sequence of rows, each row a sequence of numbers, *all with the same
        length* (a rectangular matrix).
    vector:
        Sequence of numbers whose length equals the row length.
    allow_nonfinite:
        Forwarded to :func:`dot_product`.

    Returns
    -------
    list
        One number per row of ``matrix``; ``[]`` for an empty matrix.

    Raises
    ------
    TypeError
        If ``matrix`` is not a sequence of sequences, or a row/entry is not a
        number (a flat 1-D list like ``[1, 2, 3]`` is a ``TypeError``, not a
        misunderstood matrix).
    ValueError
        If rows have unequal lengths (ragged), if row length != ``len(vector)``,
        or if a non-finite value appears while ``allow_nonfinite`` is ``False``.
    """
    rows_of_raw = _as_list("matrix", matrix)
    vector_list = _as_list("vector", vector)

    # Validate the vector once here (not once per row) for speed, then let
    # dot_product do the numeric checks again on the combined inputs.
    _check_numbers("vector", vector_list, allow_nonfinite)

    rows: List[List[Number]] = []
    expected_width: int | None = None
    for row_index, row in enumerate(rows_of_raw):
        if isinstance(row, _STRING_LIKE) or isinstance(row, Number):
            raise TypeError(
                f"matrix[{row_index}] must be a sequence of numbers, "
                f"got {type(row).__name__}; matrix must be a list of rows "
                f"(e.g. [[1, 2], [3, 4]]), not a flat list of numbers"
            )
        try:
            row_list = list(row)
        except TypeError as exc:
            raise TypeError(
                f"matrix[{row_index}] must be an iterable sequence of numbers, "
                f"got {type(row).__name__}"
            ) from exc

        # Rectangularity: ragged rows would make some dot products silently
        # raise later, or worse, be padded by accident in some other backend.
        if expected_width is None:
            expected_width = len(row_list)
        elif len(row_list) != expected_width:
            raise ValueError(
                f"ragged matrix: matrix[{row_index}] has {len(row_list)} entries, "
                f"expected {expected_width} like the first row"
            )

        rows.append(row_list)

    # Row length must equal len(vector): A @ x is only defined for x of width(A).
    if expected_width is not None and expected_width != len(vector_list):
        raise ValueError(
            f"dimension mismatch: each matrix row has {expected_width} entries "
            f"but len(vector)={len(vector_list)}"
        )
    if expected_width is None and vector_list:
        # An empty matrix can only legitimately multiply an empty vector.
        raise ValueError(
            f"dimension mismatch: matrix has no rows but len(vector)={len(vector_list)}"
        )

    # Each entry reuses dot_product, so validation and the loop live in one place.
    return [dot_product(row, vector_list, allow_nonfinite=allow_nonfinite) for row in rows]


def _reference_matvec(
    matrix: Sequence[Sequence[Number]], vector: Sequence[Number]
) -> List[Number]:
    """Independent reference implementation used only to cross-check results.

    Deliberately written differently (``sum`` over ``zip``, no validation) so a
    bug in ``matvec_multiply`` is unlikely to be reproduced here.
    """
    return [sum(a_ij * x_j for a_ij, x_j in zip(row, vector)) for row in matrix]


def main(n_rows: int = 1000, n_cols: int = 1000, seed: int | None = 0) -> List[Number]:
    """Test :func:`matvec_multiply` on randomly generated ``n_rows x n_cols`` data.

    Steps: build random data with a fixed seed (so runs are reproducible),
    multiply, then verify the result against the reference implementation,
    its expected shape, and finiteness of every entry.  Prints a short report.
    """
    rng = random.Random(seed)

    matrix = [[rng.uniform(-1.0, 1.0) for _ in range(n_cols)] for _ in range(n_rows)]
    vector = [rng.uniform(-1.0, 1.0) for _ in range(n_cols)]

    start = time.perf_counter()
    result = matvec_multiply(matrix, vector)
    elapsed = time.perf_counter() - start

    # --- assertions: fail loudly rather than return a plausible wrong answer --
    assert len(result) == n_rows, f"expected {n_rows} entries, got {len(result)}"
    assert all(math.isfinite(v) for v in result), "result contains NaN/inf"

    reference = _reference_matvec(matrix, vector)
    max_diff = max(
        (abs(got - want) for got, want in zip(result, reference)), default=0.0
    )
    assert max_diff < 1e-9, f"max deviation from reference too large: {max_diff}"

    print(f"matvec_multiply: {n_rows}x{n_cols} matrix-vector product OK")
    print(f"  elapsed      : {elapsed:.3f} s ({n_rows * n_cols} multiply-adds)")
    print(f"  max deviation: {max_diff:.3e} (reference cross-check)")
    print(f"  result[0]    : {result[0]:.6f}")
    return result


if __name__ == "__main__":
    # Allow overrides from the command line, e.g.
    #   python matvec_multiply.py 100 100 42
    import sys

    args = sys.argv[1:]
    main(
        n_rows=int(args[0]) if len(args) > 0 else 1000,
        n_cols=int(args[1]) if len(args) > 1 else 1000,
        seed=int(args[2]) if len(args) > 2 else 0,
    )
