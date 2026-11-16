"""main.py 단위 테스트 (표준 라이브러리 unittest).

실행: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import main  # noqa: E402


def cross_rows(n):
    c = n // 2
    return [[1 if (r == c or col == c) else 0 for col in range(n)] for r in range(n)]


def x_rows(n):
    return [[1 if (r == col or r + col == n - 1) else 0 for col in range(n)] for r in range(n)]


class MatrixTest(unittest.TestCase):
    def test_new_matrix_is_zero_filled(self):
        m = main.Matrix(3)
        self.assertEqual(m.to_rows(), [[0.0] * 3] * 3)

    def test_set_and_get(self):
        m = main.Matrix(3)
        m.set(1, 2, 7)
        self.assertEqual(m.get(1, 2), 7.0)
        self.assertIsInstance(m.get(1, 2), float)

    def test_out_of_range_index(self):
        m = main.Matrix(3)
        with self.assertRaises(IndexError):
            m.get(3, 0)
        with self.assertRaises(IndexError):
            m.set(0, -1, 1)

    def test_required_sizes(self):
        for n in (3, 5, 13, 25):
            m = main.Matrix.from_rows(cross_rows(n))
            self.assertEqual(m.n, n)
            self.assertEqual(m.get(n // 2, 0), 1.0)

    def test_to_rows_returns_copy(self):
        m = main.Matrix.from_rows(cross_rows(3))
        rows = m.to_rows()
        rows[0][0] = 99
        self.assertEqual(m.get(0, 0), 0.0)

    def test_invalid_rows(self):
        bad_inputs = [
            [],                          # 빈 리스트
            "abc",                       # 리스트가 아님
            [[1, 2], [3]],               # 행 길이 불일치
            [[1, 2, 3], [4, 5, 6]],      # 2×3 (정사각형 아님)
            [[1, "a"], [1, 1]],          # 숫자가 아닌 값
            [[True, 0], [0, 0]],         # bool
            [[float("nan")]],            # NaN
            [[float("inf")]],            # inf
        ]
        for rows in bad_inputs:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    main.Matrix.from_rows(rows)

    def test_invalid_size(self):
        for n in (0, -1, 2.0, True):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    main.Matrix(n)


class MacTest(unittest.TestCase):
    def test_mission_example_3x3(self):
        cross = main.Matrix.from_rows(cross_rows(3))
        x = main.Matrix.from_rows(x_rows(3))
        self.assertEqual(main.mac(cross, cross), 5.0)  # Case 1: 일치
        self.assertEqual(main.mac(cross, x), 1.0)      # Case 2: 불일치
        self.assertEqual(main.mac(x, x), 5.0)

    def test_scores_scale_with_size(self):
        # N×N 십자가는 1이 2N-1개, X는 2N-1개, 둘이 겹치는 곳은 중앙 1칸
        for n in (5, 13, 25):
            cross = main.Matrix.from_rows(cross_rows(n))
            x = main.Matrix.from_rows(x_rows(n))
            self.assertEqual(main.mac(cross, cross), 2 * n - 1)
            self.assertEqual(main.mac(cross, x), 1.0)

    def test_returns_float_with_fractional_weights(self):
        p = main.Matrix.from_rows([[1, 1], [1, 1]])
        f = main.Matrix.from_rows([[0.5, 0.25], [0, 1.5]])
        score = main.mac(p, f)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 2.25)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            main.mac(main.Matrix(3), main.Matrix(5))


if __name__ == "__main__":
    unittest.main()
