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


class NormalizeLabelTest(unittest.TestCase):
    def test_expected_values(self):
        self.assertEqual(main.normalize_label("+"), main.CROSS)
        self.assertEqual(main.normalize_label("x"), main.X)

    def test_filter_keys(self):
        self.assertEqual(main.normalize_label("cross"), main.CROSS)
        self.assertEqual(main.normalize_label("x"), main.X)

    def test_case_and_whitespace_insensitive(self):
        for raw in ("Cross", "CROSS", " cross "):
            self.assertEqual(main.normalize_label(raw), main.CROSS)
        for raw in ("X", " x "):
            self.assertEqual(main.normalize_label(raw), main.X)

    def test_unknown_values(self):
        for raw in ("o", "", "plus", None, 1, ["x"]):
            with self.subTest(raw=raw):
                self.assertIsNone(main.normalize_label(raw))


class JudgeTest(unittest.TestCase):
    def test_clear_winner(self):
        self.assertEqual(main.judge(5.0, 1.0, "A", "B"), "A")
        self.assertEqual(main.judge(1.0, 5.0, "A", "B"), "B")

    def test_exact_tie(self):
        self.assertEqual(main.judge(1.0, 1.0, "A", "B"), main.UNDECIDED)

    def test_floating_point_tie(self):
        # 0.1 + 0.2 = 0.30000000000000004 이므로 '>'만 쓰면 A가 이긴다.
        self.assertTrue(0.1 + 0.2 > 0.3)
        self.assertEqual(main.judge(0.1 + 0.2, 0.3, "A", "B"), main.UNDECIDED)

    def test_epsilon_boundary(self):
        self.assertEqual(main.judge(1.0, 1.0 + 1e-10, "A", "B"), main.UNDECIDED)
        self.assertEqual(main.judge(1.0, 1.0 + 1e-8, "A", "B"), "B")


class DataJsonScoreTest(unittest.TestCase):
    """제공된 data.json의 점수가 예상대로 계산·판정되는지 확인한다."""

    # case -> (판정, expected)
    EXPECTED = {
        "size_5_1": (main.UNDECIDED, main.X),
        "size_5_2": (main.CROSS, main.CROSS),
        "size_13_1": (main.X, main.X),
        "size_13_2": (main.UNDECIDED, main.CROSS),
        "size_25_1": (main.UNDECIDED, main.X),
        "size_25_2": (main.CROSS, main.CROSS),
    }

    @classmethod
    def setUpClass(cls):
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data.json")
        with open(path, encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_judgements(self):
        filters = {
            key: {main.normalize_label(label): main.Matrix.from_rows(rows)
                  for label, rows in pair.items()}
            for key, pair in self.data["filters"].items()
        }
        for key, (want_decision, want_expected) in self.EXPECTED.items():
            with self.subTest(case=key):
                entry = self.data["patterns"][key]
                n = int(key.split("_")[1])
                pattern = main.Matrix.from_rows(entry["input"])
                pair = filters["size_{}".format(n)]
                score_cross = main.mac(pattern, pair[main.CROSS])
                score_x = main.mac(pattern, pair[main.X])
                self.assertEqual(
                    main.judge(score_cross, score_x, main.CROSS, main.X), want_decision
                )
                self.assertEqual(main.normalize_label(entry["expected"]), want_expected)


def fake_input(lines):
    """input()을 대신해 미리 정한 줄을 차례로 돌려준다. 다 쓰면 EOFError."""
    iterator = iter(lines)

    def _input(prompt=""):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError
    return _input


def run_quietly(func, *args):
    """함수를 실행하고 (반환값, 출력 문자열)을 돌려준다."""
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args)
    return result, buffer.getvalue()


class ParseRowTest(unittest.TestCase):
    def test_valid_row(self):
        self.assertEqual(main.parse_row("0 1 0", 3), [0.0, 1.0, 0.0])
        self.assertEqual(main.parse_row("  0.5\t-1   2 ", 3), [0.5, -1.0, 2.0])

    def test_wrong_count(self):
        for line in ("", "0 1", "0 1 0 1", "0,1,0"):
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, "각 줄에 3개의 숫자를 공백으로 구분해"):
                    main.parse_row(line, 3)

    def test_not_a_number(self):
        for line in ("0 a 0", "0 nan 0", "inf 0 0"):
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, "입력 형식 오류"):
                    main.parse_row(line, 3)


class ReadMatrixTest(unittest.TestCase):
    def test_retries_only_bad_lines(self):
        lines = ["0 1 0", "1 1", "x 1 1", "1 1 1", "0 1 0"]
        matrix, output = run_quietly(main.read_matrix, 3, "패턴", fake_input(lines))
        self.assertEqual(matrix.to_rows(), [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(output.count("2번째 줄을 다시 입력하세요"), 2)

    def test_eof_propagates(self):
        with self.assertRaises(EOFError):
            run_quietly(main.read_matrix, 3, "패턴", fake_input(["0 1 0"]))


class ChooseModeTest(unittest.TestCase):
    def test_retries_until_valid(self):
        choice, output = run_quietly(main.choose_mode, fake_input(["3", "", " 2 "]))
        self.assertEqual(choice, "2")
        self.assertEqual(output.count("1 또는 2를 입력하세요"), 2)


class UserModeTest(unittest.TestCase):
    CROSS_3 = ["0 1 0", "1 1 1", "0 1 0"]
    X_3 = ["1 0 1", "0 1 0", "1 0 1"]

    def test_mission_example(self):
        lines = self.CROSS_3 + self.X_3 + self.X_3
        decision, output = run_quietly(main.run_user_mode, fake_input(lines))
        self.assertEqual(decision, "B")
        self.assertIn("A 점수: 1.0", output)
        self.assertIn("B 점수: 5.0", output)
        self.assertIn("판정: B", output)
        self.assertIn("3×3", output)

    def test_floating_point_tie_is_undecided(self):
        lines = ["0 0.1 0", "0 0.2 0", "0 0 0",
                 "0.3 0 0", "0 0 0", "0 0 0",
                 "1 1 1", "1 1 1", "1 1 1"]
        decision, output = run_quietly(main.run_user_mode, fake_input(lines))
        self.assertEqual(decision, main.UNDECIDED)
        self.assertIn("A 점수: 0.30000000000000004", output)
        self.assertIn("판정 불가 (|A-B| < 1e-9)", output)


class MeasureTest(unittest.TestCase):
    def test_calls_warmup_plus_repeat(self):
        calls = []
        avg = main.measure_average_ms(lambda: calls.append(1), repeat=10)
        self.assertEqual(len(calls), 11)
        self.assertGreaterEqual(avg, 0.0)


PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ERROR_CASES_PATH = os.path.join(PROJECT_DIR, "samples", "error_cases.json")


class LoadJsonTest(unittest.TestCase):
    def _write(self, text):
        import tempfile
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.remove, handle.name)
        return handle.name

    def test_valid_file(self):
        data, error = main.load_json(main.DEFAULT_DATA_PATH)
        self.assertIsNone(error)
        self.assertIn("patterns", data)

    def test_missing_file(self):
        data, error = main.load_json("does_not_exist.json")
        self.assertIsNone(data)
        self.assertIn("찾을 수 없습니다", error)

    def test_syntax_error(self):
        data, error = main.load_json(self._write('{"a": 1,'))
        self.assertIsNone(data)
        self.assertIn("JSON 문법 오류", error)

    def test_top_level_not_object(self):
        data, error = main.load_json(self._write("[1, 2]"))
        self.assertIsNone(data)
        self.assertIn("객체", error)


class LoadFiltersTest(unittest.TestCase):
    def test_data_json_filters(self):
        data, _ = main.load_json(main.DEFAULT_DATA_PATH)
        filters, logs = main.load_filters(data["filters"])
        self.assertEqual(sorted(filters), [5, 13, 25])
        self.assertTrue(all(ok for ok, _, _ in logs))
        for n, pair in filters.items():
            self.assertEqual(set(pair), {main.CROSS, main.X})  # 키 정규화
            self.assertEqual(pair[main.CROSS].n, n)

    def test_invalid_filters_are_skipped_with_reason(self):
        section = {
            "size_3": {"cross": cross_rows(3), "o": x_rows(3)},   # 알 수 없는 키, X 없음
            "size_4": {"cross": cross_rows(4), "x": x_rows(3)},   # 크기 불일치
            "size_5": {"x": x_rows(5), "X": x_rows(5)},           # 중복, Cross 없음
            "big": {"cross": cross_rows(3), "x": x_rows(3)},      # 키 형식
            "size_6": [1, 2],                                     # 객체 아님
        }
        filters, logs = main.load_filters(section)
        self.assertEqual(filters, {})
        reasons = {key: message for ok, key, message in logs if not ok}
        self.assertIn("알 수 없는 필터 키", reasons["size_3"])
        self.assertIn("크기 불일치", reasons["size_4"])
        self.assertIn("중복", reasons["size_5"])
        self.assertIn("Cross 필터 없음", reasons["size_5"])
        self.assertIn("키 형식 오류", reasons["big"])
        self.assertIn("객체가 아닙니다", reasons["size_6"])

    def test_missing_section(self):
        filters, logs = main.load_filters(None)
        self.assertEqual(filters, {})
        self.assertFalse(logs[0][0])


class EvaluateCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filters = {5: {main.CROSS: main.Matrix.from_rows(cross_rows(5)),
                           main.X: main.Matrix.from_rows(x_rows(5))}}

    def evaluate(self, key, entry):
        return main.evaluate_case(key, entry, self.filters)

    def test_pass(self):
        result = self.evaluate("size_5_1", {"input": cross_rows(5), "expected": "+"})
        self.assertTrue(result.passed)
        self.assertEqual(result.decision, main.CROSS)
        self.assertEqual((result.score_cross, result.score_x), (9.0, 1.0))
        self.assertIsNone(result.reason)

    def assertFailCategory(self, key, entry, category):
        result = self.evaluate(key, entry)
        self.assertFalse(result.passed)
        self.assertTrue(result.reason.startswith(category), result.reason)
        return result

    def test_fail_categories(self):
        cases = [
            ("pattern_1", {"input": cross_rows(5), "expected": "+"}, main.REASON_SCHEMA),
            ("size_5_1", [1, 2], main.REASON_SCHEMA),
            ("size_5_1", {"input": cross_rows(5)}, main.REASON_SCHEMA),
            ("size_7_1", {"input": cross_rows(7), "expected": "+"}, main.REASON_SCHEMA),
            ("size_5_1", {"input": [[0, 1]] * 5, "expected": "+"}, main.REASON_SCHEMA),
            ("size_5_1", {"input": cross_rows(4), "expected": "+"}, main.REASON_SIZE),
            ("size_5_1", {"input": cross_rows(5), "expected": "o"}, main.REASON_LABEL),
            ("size_5_1", {"input": cross_rows(5), "expected": "x"}, main.REASON_MISMATCH),
        ]
        for key, entry, category in cases:
            with self.subTest(key=key, category=category, entry=entry):
                self.assertFailCategory(key, entry, category)

    def test_tie_fails_with_undecided(self):
        center = [[1 if (r == 2 and c == 2) else 0 for c in range(5)] for r in range(5)]
        result = self.assertFailCategory(
            "size_5_1", {"input": center, "expected": "+"}, main.REASON_TIE
        )
        self.assertEqual(result.decision, main.UNDECIDED)

    def test_label_failure_still_reports_scores(self):
        result = self.evaluate("size_5_1", {"input": cross_rows(5), "expected": "o"})
        self.assertEqual(result.score_cross, 9.0)


class RunJsonModeTest(unittest.TestCase):
    def test_data_json(self):
        results, output = run_quietly(main.run_json_mode, main.DEFAULT_DATA_PATH)
        self.assertEqual(len(results), 6)
        self.assertEqual(sum(r.passed for r in results), 3)
        failed = {r.key for r in results if not r.passed}
        self.assertEqual(failed, {"size_5_1", "size_13_2", "size_25_1"})
        self.assertTrue(all(r.reason.startswith(main.REASON_TIE) for r in results if not r.passed))
        self.assertIn("✓ size_25", output)

    def test_error_cases_do_not_stop_program(self):
        results, output = run_quietly(main.run_json_mode, ERROR_CASES_PATH)
        self.assertEqual(len(results), 14)
        self.assertEqual([r.key for r in results if r.passed], ["size_5_1", "size_5_2"])
        self.assertIn("✗ size_9", output)

    def test_unreadable_file_returns_none(self):
        result, output = run_quietly(main.run_json_mode, "does_not_exist.json")
        self.assertIsNone(result)
        self.assertIn("불러오지 못했습니다", output)


class MainTest(unittest.TestCase):
    def run_main(self, argv, lines):
        from unittest import mock
        with mock.patch("builtins.input", fake_input(lines)), \
                mock.patch.object(main, "configure_output"):
            return run_quietly(main.main, argv)

    def test_mode2_with_path_argument(self):
        code, output = self.run_main([ERROR_CASES_PATH], ["2"])
        self.assertEqual(code, 0)
        self.assertIn("pattern_1", output)

    def test_mode2_missing_file(self):
        code, output = self.run_main(["does_not_exist.json"], ["2"])
        self.assertEqual(code, 1)
        self.assertIn("불러오지 못했습니다", output)

    def test_mode1_via_main(self):
        lines = ["1"] + UserModeTest.CROSS_3 + UserModeTest.X_3 + UserModeTest.CROSS_3
        code, output = self.run_main([], lines)
        self.assertEqual(code, 0)
        self.assertIn("판정: A", output)

    def test_eof_exits_cleanly(self):
        code, output = self.run_main([], [])
        self.assertEqual(code, 1)
        self.assertIn("입력이 종료되어", output)


if __name__ == "__main__":
    unittest.main()
