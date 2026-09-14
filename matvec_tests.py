"""Unit tests for matvec_multiply.py."""

import math
import random
import unittest

from matvec_multiply import dot_product, matvec_multiply


class DotProductTests(unittest.TestCase):
    def test_integer_vectors(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)

    def test_negative_and_float_values(self):
        self.assertAlmostEqual(dot_product([-1.5, 2.0], [2.0, 4.0]), 5.0)

    def test_empty_vectors(self):
        self.assertEqual(dot_product([], []), 0)

    def test_generators(self):
        self.assertEqual(dot_product((x for x in [1, 2]), (x for x in [3, 4])), 11)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            dot_product([1, 2], [1])

    def test_non_iterable_input(self):
        with self.assertRaises(TypeError):
            dot_product(3, [1, 2, 3])

    def test_text_input(self):
        with self.assertRaises(TypeError):
            dot_product("123", [1, 2, 3])

    def test_non_numeric_member(self):
        with self.assertRaises(TypeError):
            dot_product([1, "2"], [3, 4])

    def test_boolean_member(self):
        with self.assertRaises(TypeError):
            dot_product([True, 2], [3, 4])

    def test_nan_and_infinity(self):
        for invalid_value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    dot_product([invalid_value], [1.0])


class MatrixVectorTests(unittest.TestCase):
    def test_known_result(self):
        matrix = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(matvec_multiply(matrix, [7, 8, 9]), [50, 122])

    def test_identity_matrix(self):
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(matvec_multiply(matrix, [3, -2, 5]), [3, -2, 5])

    def test_rectangular_matrix(self):
        self.assertEqual(
            matvec_multiply([[1, 2], [3, 4], [5, 6]], [2, 1]),
            [4, 10, 16],
        )

    def test_empty_matrix(self):
        self.assertEqual(matvec_multiply([], []), [])

    def test_empty_rows_and_vector(self):
        self.assertEqual(matvec_multiply([[], []], []), [0, 0])

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            matvec_multiply([[1, 2], [3, 4]], [1, 2, 3])

    def test_ragged_matrix(self):
        with self.assertRaises(ValueError):
            matvec_multiply([[1, 2], [3]], [1, 2])

    def test_flat_list_is_not_a_matrix(self):
        with self.assertRaises(TypeError):
            matvec_multiply([1, 2], [3, 4])

    def test_invalid_matrix_entry(self):
        with self.assertRaises(TypeError):
            matvec_multiply([[1, "bad"]], [1, 2])

    def test_inputs_are_not_changed(self):
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        matvec_multiply(matrix, vector)
        self.assertEqual(matrix, [[1, 2], [3, 4]])
        self.assertEqual(vector, [5, 6])

    def test_random_values_against_reference(self):
        generator = random.Random(42)
        matrix = [
            [generator.uniform(-10, 10) for _ in range(20)] for _ in range(15)
        ]
        vector = [generator.uniform(-10, 10) for _ in range(20)]
        expected = [
            sum(value * vector[j] for j, value in enumerate(row))
            for row in matrix
        ]
        actual = matvec_multiply(matrix, vector)
        for calculated, reference in zip(actual, expected):
            self.assertAlmostEqual(calculated, reference, places=10)


if __name__ == "__main__":
    unittest.main()
