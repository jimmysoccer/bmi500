"""Tests for :mod:`matvec_multiply`.

Written in pytest style (plain ``test_*`` functions + ``assert``), but with a
tiny built-in runner so it also works with no third-party packages installed:

    python matvec_tests.py          # no pytest needed
    python -m pytest matvec_tests.py   # also fine if pytest is installed

The tests are organised in two blocks, mirroring the two functions required by
the assignment:

1. ``dot_product``      -- correctness + every guarded failure mode.
2. ``matvec_multiply``  -- correctness + shape/validation failure modes.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from numbers import Number

from matvec_multiply import dot_product, matvec_multiply

# Tolerance for floating-point comparisons. Values below are O(1..100), so an
# absolute tolerance of 1e-9 is strict enough to catch a wrong algorithm but
# loose enough to survive float rounding-order differences.
TOL = 1e-9


@contextmanager
def raises(exc_type):
    """Minimal stand-in for ``pytest.raises`` so the file runs without pytest."""
    try:
        yield
    except exc_type:
        return
    except Exception as caught:  # wrong exception type => failure
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(caught).__name__}: {caught}"
        ) from caught
    raise AssertionError(f"expected {exc_type.__name__}, but nothing was raised")


def _close(a, b):
    return abs(a - b) < TOL


# ---------------------------------------------------------------------------
# 1. dot_product
# ---------------------------------------------------------------------------


def test_dot_product_basic_integers():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_dot_product_known_float_values():
    assert _close(dot_product([0.5, -1.5, 2.0], [4.0, 2.0, -1.0]), -3.0)


def test_dot_product_mixed_int_and_float_returns_float():
    result = dot_product([1, 2], [0.5, 0.25])
    assert isinstance(result, float)
    assert _close(result, 1.0)


def test_dot_product_int_only_stays_exact_int():
    result = dot_product([10, 20], [3, 4])
    assert isinstance(result, int)
    assert result == 110


def test_dot_product_with_zeros():
    assert dot_product([0, 0, 0], [1, 2, 3]) == 0
    assert dot_product([1, 2, 3], [0, 0, 0]) == 0


def test_dot_product_negative_cancellation():
    assert dot_product([1, -1], [1, 1]) == 0


def test_dot_product_single_element():
    assert dot_product([7], [3]) == 21


def test_dot_product_empty_vectors_is_zero():
    # The empty sum is 0 -- this is the mathematically correct convention.
    assert dot_product([], []) == 0


def test_dot_product_accepts_tuples_and_ranges():
    assert dot_product((1, 2, 3), (4, 5, 6)) == 32
    assert dot_product(range(1, 4), [4, 5, 6]) == 32


def test_dot_product_accepts_generators():
    assert dot_product((i for i in [1, 2, 3]), (i for i in [4, 5, 6])) == 32


def test_dot_product_is_commutative():
    a = [1.5, -2.0, 3.25]
    b = [-4.0, 0.5, 2.0]
    assert _close(dot_product(a, b), dot_product(b, a))


def test_dot_product_length_mismatch_raises():
    with raises(ValueError):
        dot_product([1, 2, 3], [1, 2])
    with raises(ValueError):
        dot_product([1, 2], [1, 2, 3])


def test_dot_product_rejects_string_argument():
    # Strings are iterable, so without a guard this would silently iterate chars.
    with raises(TypeError):
        dot_product("abc", [1, 2, 3])
    with raises(TypeError):
        dot_product([1, 2, 3], "abc")


def test_dot_product_rejects_bare_scalar():
    with raises(TypeError):
        dot_product(5, [1, 2, 3])
    with raises(TypeError):
        dot_product([1, 2, 3], None)


def test_dot_product_rejects_non_numeric_entries():
    # "3" * 2 succeeds in Python, so this would silently concatenate strings.
    with raises(TypeError):
        dot_product([1, "3"], [1, 2])
    with raises(TypeError):
        dot_product([1, 2], [1, None])


def test_dot_product_rejects_booleans():
    # bool is a subclass of int; True * True == 1 would hide upstream logic bugs.
    with raises(TypeError):
        dot_product([True, False], [1, 1])


def test_dot_product_rejects_nested_lists():
    with raises(TypeError):
        dot_product([[1, 2], [3, 4]], [1, 1])


def test_dot_product_nonfinite_guard():
    with raises(ValueError):
        dot_product([1.0, float("nan")], [1.0, 1.0])
    with raises(ValueError):
        dot_product([1.0, float("inf")], [1.0, 1.0])


def test_dot_product_allows_nonfinite_on_request():
    assert math.isinf(dot_product([1.0, float("inf")], [1.0, 2.0], allow_nonfinite=True))
    assert math.isnan(dot_product([1.0, float("nan")], [1.0, 2.0], allow_nonfinite=True))


def test_dot_product_input_not_mutated():
    a, b = [1, 2, 3], [4, 5, 6]
    dot_product(a, b)
    assert a == [1, 2, 3] and b == [4, 5, 6]


def test_dot_product_matches_random_reference():
    rng = random.Random(1234)
    for _ in range(50):
        n = rng.randint(1, 40)
        a = [rng.uniform(-10, 10) for _ in range(n)]
        b = [rng.uniform(-10, 10) for _ in range(n)]
        expected = sum(ai * bi for ai, bi in zip(a, b))
        assert _close(dot_product(a, b), expected)


# ---------------------------------------------------------------------------
# 2. matvec_multiply
# ---------------------------------------------------------------------------


def test_matvec_known_2x3_example():
    matrix = [[1, 2, 3], [4, 5, 6]]
    vector = [7, 8, 9]
    # 1*7 + 2*8 + 3*9 = 50 ; 4*7 + 5*8 + 6*9 = 122
    assert matvec_multiply(matrix, vector) == [50, 122]


def test_matvec_results_are_row_dot_products():
    matrix = [[1.5, -2.0], [0.0, 3.0], [-1.0, -1.0]]
    vector = [2.0, 4.0]
    for row, got in zip(matrix, matvec_multiply(matrix, vector)):
        assert _close(got, dot_product(row, vector))


def test_matvec_identity_returns_vector():
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    vector = [3, -4, 5]
    assert matvec_multiply(identity, vector) == vector


def test_matvec_zero_matrix_gives_zeros():
    matrix = [[0, 0], [0, 0]]
    assert matvec_multiply(matrix, [1, 2]) == [0, 0]


def test_matvec_output_length_equals_row_count():
    matrix = [[1, 2]] * 7
    assert len(matvec_multiply(matrix, [1, 1])) == 7


def test_matvec_single_row_and_single_column():
    assert matvec_multiply([[2, 3]], [4, 5]) == [23]
    assert matvec_multiply([[1], [2], [3]], [10]) == [10, 20, 30]


def test_matvec_matches_manual_computation():
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    vector = [-1, 2, -3, 4]
    expected = [
        1 * -1 + 2 * 2 + 3 * -3 + 4 * 4,
        5 * -1 + 6 * 2 + 7 * -3 + 8 * 4,
        9 * -1 + 10 * 2 + 11 * -3 + 12 * 4,
    ]
    got = matvec_multiply(matrix, vector)
    assert len(got) == len(expected)
    assert all(_close(g, e) for g, e in zip(got, expected))


def test_matvec_is_linear():
    matrix = [[1, 2, 3], [4, 5, 6]]
    x = [1.0, -2.0, 3.0]
    y = [0.5, 1.5, -1.0]
    lhs = matvec_multiply(matrix, [xi + yi for xi, yi in zip(x, y)])
    rhs = [
        a + b
        for a, b in zip(matvec_multiply(matrix, x), matvec_multiply(matrix, y))
    ]
    assert all(_close(p, q) for p, q in zip(lhs, rhs))


def test_matvec_row_length_mismatch_raises():
    with raises(ValueError):
        matvec_multiply([[1, 2, 3], [4, 5, 6]], [1, 2])


def test_matvec_ragged_matrix_raises():
    # A ragged matrix has no well-defined column count.
    with raises(ValueError):
        matvec_multiply([[1, 2, 3], [4, 5]], [1, 1, 1])


def test_matvec_flat_list_is_rejected():
    # Common mistake: passing a 1-D list where a list of rows is expected.
    with raises(TypeError):
        matvec_multiply([1, 2, 3], [1, 1, 1])


def test_matvec_rejects_string_matrix_or_row():
    with raises(TypeError):
        matvec_multiply("abc", [1, 1, 1])
    with raises(TypeError):
        matvec_multiply([["a", "b"], ["c", "d"]], [1, 1])


def test_matvec_rejects_non_numeric_vector_entries():
    with raises(TypeError):
        matvec_multiply([[1, 2], [3, 4]], ["1", 2])


def test_matvec_empty_matrix_and_empty_vector():
    assert matvec_multiply([], []) == []


def test_matvec_empty_matrix_with_nonempty_vector_raises():
    with raises(ValueError):
        matvec_multiply([], [1, 2])


def test_matvec_nonfinite_propagates_as_error():
    with raises(ValueError):
        matvec_multiply([[1.0, float("nan")]], [1.0, 1.0])
    with raises(ValueError):
        matvec_multiply([[1.0, 2.0]], [1.0, float("inf")])


def test_matvec_large_random_against_reference():
    """Stress test: 200x300 against an independent implementation."""
    rng = random.Random(7)
    rows, cols = 200, 300
    matrix = [[rng.uniform(-5, 5) for _ in range(cols)] for _ in range(rows)]
    vector = [rng.uniform(-5, 5) for _ in range(cols)]

    got = matvec_multiply(matrix, vector)
    assert len(got) == rows
    assert all(isinstance(v, Number) and math.isfinite(v) for v in got)

    expected = [sum(a_ij * x_j for a_ij, x_j in zip(row, vector)) for row in matrix]
    worst = max(abs(g - e) for g, e in zip(got, expected))
    assert worst < 1e-6, f"max deviation too large: {worst}"


def test_matvec_input_not_mutated():
    matrix = [[1, 2], [3, 4]]
    vector = [5, 6]
    matvec_multiply(matrix, vector)
    assert matrix == [[1, 2], [3, 4]] and vector == [5, 6]


# ---------------------------------------------------------------------------
# Runner (used when pytest is unavailable)
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures = []
    for name, func in tests:
        try:
            func()
        except Exception as exc:  # noqa: BLE001 - report everything
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print("-" * 60)
    print(f"{len(tests) - len(failures)} passed, {len(failures)} failed, {len(tests)} total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
