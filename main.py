"""Mini NPU Simulator.

MAC(Multiply-Accumulate) 연산으로 N×N 패턴이 Cross(십자가)인지 X인지 판별한다.
외부 라이브러리 없이 Python 표준 라이브러리만 사용한다.
"""

import json
import math
import os
import re
import sys
import time
import unicodedata


# ============================================================
# 상수: 표준 라벨
# ============================================================

CROSS = "Cross"
X = "X"
UNDECIDED = "UNDECIDED"

# 점수 비교 허용오차: |A - B| < EPSILON 이면 동점(UNDECIDED)으로 본다.
EPSILON = 1e-9

# 성능 측정 반복 횟수 (과제 최소 기준: 10회)
REPEAT = 10


# ============================================================
# 데이터 구조: N×N 행렬
# ============================================================

def _to_number(value):
    """행렬에 저장할 값을 float로 바꾼다. 숫자가 아니거나 유한하지 않으면 ValueError."""
    # bool은 int의 하위 타입이라 True/False가 1/0으로 통과하는 것을 막는다.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("숫자가 아닌 값이 있습니다: {!r}".format(value))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("유한한 숫자가 아닙니다: {!r}".format(value))
    return number


def describe_shape(rows):
    """2차원 리스트의 모양을 사람이 읽기 쉬운 문자열로 설명한다 (오류 메시지용)."""
    if not isinstance(rows, list):
        return "리스트가 아님({})".format(type(rows).__name__)
    if not rows:
        return "빈 리스트"
    lengths = [len(r) if isinstance(r, list) else None for r in rows]
    if None in lengths:
        return "{}행 (리스트가 아닌 행 포함)".format(len(rows))
    if len(set(lengths)) == 1:
        return "{}×{}".format(len(rows), lengths[0])
    return "{}행, 행 길이 불일치 {}".format(len(rows), lengths)


class Matrix:
    """N×N 정사각 행렬. 패턴과 필터를 모두 이 구조로 저장한다."""

    def __init__(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("행렬 크기는 1 이상의 정수여야 합니다: {!r}".format(n))
        self.n = n
        self._cells = [[0.0] * n for _ in range(n)]

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트로부터 행렬을 만든다. 정사각형이 아니거나 숫자가 아니면 ValueError."""
        if not isinstance(rows, list) or not rows:
            raise ValueError("2차원 리스트가 아닙니다: {}".format(describe_shape(rows)))
        n = len(rows)
        for row in rows:
            if not isinstance(row, list) or len(row) != n:
                raise ValueError("정사각 행렬이 아닙니다: {}".format(describe_shape(rows)))
        matrix = cls(n)
        for r in range(n):
            for c in range(n):
                matrix.set(r, c, rows[r][c])
        return matrix

    def _check_index(self, row, col):
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise IndexError(
                "위치 ({}, {})는 {}×{} 행렬 범위를 벗어납니다".format(row, col, self.n, self.n)
            )

    def get(self, row, col):
        """(row, col) 위치의 값을 읽는다."""
        self._check_index(row, col)
        return self._cells[row][col]

    def set(self, row, col, value):
        """(row, col) 위치에 값을 저장한다."""
        self._check_index(row, col)
        self._cells[row][col] = _to_number(value)

    @property
    def cells(self):
        """내부 2차원 리스트 (연산용, 읽기 전용으로 사용)."""
        return self._cells

    def to_rows(self):
        """내부 데이터의 복사본을 2차원 리스트로 돌려준다."""
        return [row[:] for row in self._cells]

    def flatten(self):
        """행 우선(row-major) 순서의 1차원 리스트(길이 N²)로 돌려준다.

        (row, col) 위치는 1차원에서 row * N + col 인덱스가 된다.
        """
        return [value for row in self._cells for value in row]

    def __repr__(self):
        return "Matrix(n={})".format(self.n)


# ============================================================
# MAC 연산 (Multiply-Accumulate)
# ============================================================

def mac(pattern, filt):
    """패턴과 필터를 같은 위치끼리 곱하고(Multiply) 모두 더한(Accumulate) 점수를 반환한다.

    벡터화 라이브러리 없이 이중 반복문으로 N×N번 곱셈과 덧셈을 수행한다.
    따라서 연산 횟수는 N², 시간 복잡도는 O(N²)이다.
    """
    if pattern.n != filt.n:
        raise ValueError(
            "크기 불일치: 패턴 {0}×{0}, 필터 {1}×{1}".format(pattern.n, filt.n)
        )
    n = pattern.n
    a = pattern.cells
    b = filt.cells
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += a[i][j] * b[i][j]
    return total


def mac_flat(a, b):
    """[보너스 최적화] 1차원 리스트(길이 N²) 두 개의 MAC.

    2차원 버전은 매번 a[i][j]처럼 '행 찾기 → 열 찾기' 두 번 인덱싱하고
    반복문도 두 겹이다. 1차원으로 펼치면 반복문 한 겹, 인덱싱 한 번으로 줄어든다.
    행 우선 순서로 펼쳤으므로 더하는 순서가 2차원 버전과 같아 결과도 비트 단위로 같다.
    연산 횟수는 그대로 N²이므로 시간 복잡도는 여전히 O(N²)이고, 상수 비용만 줄어든다.
    """
    if len(a) != len(b):
        raise ValueError("길이 불일치: {} vs {}".format(len(a), len(b)))
    total = 0.0
    for k in range(len(a)):
        total += a[k] * b[k]
    return total


# ============================================================
# 라벨 정규화
# ============================================================

# data.json은 같은 뜻을 서로 다른 표기로 쓴다.
#   expected 값 : "+" / "x"
#   filter 키   : "cross" / "x"
# 비교 전에 모두 표준 라벨(Cross / X)로 통일한다. (대소문자·앞뒤 공백 무시)
_LABEL_ALIASES = {
    "+": CROSS,
    "cross": CROSS,
    "x": X,
}


def normalize_label(raw):
    """여러 표기의 라벨을 표준 라벨(Cross/X)로 바꾼다. 해석할 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    return _LABEL_ALIASES.get(raw.strip().lower())


# ============================================================
# 판정 (허용오차 기반 비교)
# ============================================================

def is_tie(score_a, score_b, epsilon=EPSILON):
    """두 점수의 차이가 허용오차보다 작으면 동점으로 본다."""
    return abs(score_a - score_b) < epsilon


def judge(score_a, score_b, label_a, label_b, epsilon=EPSILON):
    """두 점수를 비교해 더 높은 쪽 라벨을 반환한다. 동점이면 UNDECIDED.

    부동소수점은 0.1 + 0.2 != 0.3 처럼 미세한 오차가 생긴다.
    단순히 '>'로 비교하면 사실상 같은 점수에서도 한쪽이 이긴 것처럼 판정되므로,
    차이가 epsilon보다 작으면 판정하지 않는다.
    """
    if is_tie(score_a, score_b, epsilon):
        return UNDECIDED
    return label_a if score_a > score_b else label_b


# ============================================================
# 콘솔 입력 처리 (모드 1)
# ============================================================

def parse_row(line, n):
    """한 줄 문자열을 공백으로 나눠 숫자 n개의 리스트로 바꾼다.

    개수가 맞지 않거나 숫자로 해석할 수 없으면 안내 문구를 담은 ValueError를 낸다.
    """
    tokens = line.split()
    if len(tokens) != n:
        raise ValueError(
            "입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요. "
            "(입력된 값 {}개)".format(n, len(tokens))
        )
    values = []
    for token in tokens:
        try:
            number = float(token)
        except ValueError:
            raise ValueError(
                "입력 형식 오류: '{}'은(는) 숫자가 아닙니다. "
                "각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(token, n)
            )
        if not math.isfinite(number):
            raise ValueError(
                "입력 형식 오류: '{}'은(는) 사용할 수 없는 값입니다. "
                "유한한 숫자를 입력하세요.".format(token)
            )
        values.append(number)
    return values


def read_matrix(n, title, input_fn=None):
    """n줄을 한 줄씩 입력받아 n×n Matrix를 만든다.

    잘못된 줄은 안내 문구를 보여주고 그 줄만 다시 입력받는다.
    input_fn은 테스트에서 입력을 흉내 낼 수 있도록 바꿔 끼울 수 있다.
    생략하면 호출 시점의 내장 input()을 쓴다 (기본값에 input을 고정하지 않음).
    """
    input_fn = input_fn or input
    print("{} ({}줄 입력, 공백 구분)".format(title, n))
    rows = []
    while len(rows) < n:
        line = input_fn()
        try:
            rows.append(parse_row(line, n))
        except ValueError as error:
            print("  {}".format(error))
            print("  → {}번째 줄을 다시 입력하세요.".format(len(rows) + 1))
    return Matrix.from_rows(rows)


# ============================================================
# 성능 측정
# ============================================================

def measure_average_ms(func, repeat=REPEAT):
    """func()를 repeat번 호출해 1회 평균 시간(ms)을 돌려준다.

    입출력은 측정 구간 밖에 두고, 함수 호출 구간만 time.perf_counter로 잰다.
    첫 호출은 캐시·지연 초기화 영향을 줄이기 위한 예열로 측정에서 제외한다.
    """
    func()  # 예열 (측정 제외)
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        total += time.perf_counter() - start
    return total / repeat * 1000.0


PERF_SIZES = (3, 5, 13, 25)


def print_performance_table(rows):
    """rows: (n, 평균 ms) 목록.

    크기 / 평균 시간(ms) / 연산 횟수(N²) / 연산 1회당 시간(ns)을 출력한다.
    마지막 열이 크기와 무관하게 비슷하면 시간이 N²에 비례한다는 근거가 된다.
    """
    print(
        pad("크기", 10)
        + pad("평균 시간(ms)", 16, right=True)
        + pad("연산 횟수(N²)", 16, right=True)
        + pad("연산당(ns)", 14, right=True)
    )
    print("-" * 56)
    for n, avg_ms in rows:
        per_op_ns = avg_ms * 1e6 / (n * n)
        print(
            pad("{0}×{0}".format(n), 10)
            + pad("{:.6f}".format(avg_ms), 16, right=True)
            + pad(str(n * n), 16, right=True)
            + pad("{:.1f}".format(per_op_ns), 14, right=True)
        )


# ============================================================
# 패턴 생성기
# ============================================================

def generate_cross(n):
    """N×N 십자가 패턴: 가운데 행과 가운데 열이 1. (짝수 N은 가운데 두 줄)"""
    centers = {(n - 1) // 2, n // 2}
    matrix = Matrix(n)
    for r in range(n):
        for c in range(n):
            if r in centers or c in centers:
                matrix.set(r, c, 1)
    return matrix


def generate_x(n):
    """N×N X 패턴: 두 대각선이 1."""
    matrix = Matrix(n)
    for i in range(n):
        matrix.set(i, i, 1)
        matrix.set(i, n - 1 - i, 1)
    return matrix


def measure_performance(sizes=PERF_SIZES, filters=None, repeat=REPEAT):
    """크기별 MAC 1회 평균 시간(ms)을 잰다. 결과: [(n, 평균 ms), ...]

    필터는 data.json에서 불러온 Cross 필터가 있으면 그것을, 없으면(예: 3×3)
    생성기로 만든 Cross 필터를 쓴다. 패턴은 생성기로 만든 십자가 패턴이다.
    값이 아니라 크기가 연산량을 정하므로 어떤 입력이든 N² 번 곱셈·덧셈을 한다.
    """
    return [(n, ms_2d) for n, ms_2d, _, _ in measure_comparison(sizes, filters, repeat, flat=False)]


def _perf_inputs(n, filters):
    pattern = generate_cross(n)
    filt = (filters or {}).get(n, {}).get(CROSS) or generate_cross(n)
    return pattern, filt


def measure_comparison(sizes=PERF_SIZES, filters=None, repeat=REPEAT, flat=True):
    """[보너스] 같은 입력·같은 반복 횟수로 2차원 MAC과 1차원 MAC을 비교한다.

    결과: [(n, 2D 평균 ms, 1D 평균 ms, 1차원 변환 평균 ms), ...]
    변환(flatten)은 필터를 한 번 펼쳐 두고 재사용하는 전처리이므로 MAC 시간과 따로 잰다.
    flat=False면 2차원만 잰다 (1D 관련 값은 None).
    """
    rows = []
    for n in sizes:
        pattern, filt = _perf_inputs(n, filters)
        ms_2d = measure_average_ms(lambda: mac(pattern, filt), repeat)
        if not flat:
            rows.append((n, ms_2d, None, None))
            continue
        flat_pattern, flat_filter = pattern.flatten(), filt.flatten()
        ms_1d = measure_average_ms(lambda: mac_flat(flat_pattern, flat_filter), repeat)
        ms_convert = measure_average_ms(lambda: (pattern.flatten(), filt.flatten()), repeat)
        rows.append((n, ms_2d, ms_1d, ms_convert))
    return rows


def print_comparison_table(rows):
    """[보너스] 최적화 전(2D)/후(1D) 성능 비교 표."""
    print(
        pad("크기", 10)
        + pad("2D(ms)", 12, right=True)
        + pad("1D(ms)", 12, right=True)
        + pad("속도 향상", 12, right=True)
        + pad("변환(ms)", 12, right=True)
    )
    print("-" * 58)
    for n, ms_2d, ms_1d, ms_convert in rows:
        speedup = ms_2d / ms_1d if ms_1d else float("inf")
        print(
            pad("{0}×{0}".format(n), 10)
            + pad("{:.6f}".format(ms_2d), 12, right=True)
            + pad("{:.6f}".format(ms_1d), 12, right=True)
            + pad("{:.2f}배".format(speedup), 12, right=True)
            + pad("{:.6f}".format(ms_convert), 12, right=True)
        )
    print("※ 같은 입력·같은 반복 횟수({}회). 변환(ms)은 2차원→1차원 펼치기 1회 비용(전처리, MAC 시간에 미포함)".format(REPEAT))


# ============================================================
# 출력 유틸
# ============================================================

def display_width(text):
    """콘솔 표시 폭. 한글 등 전각 문자는 2칸으로 센다."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width, right=False):
    """표시 폭 기준으로 공백을 채운다 (한글이 섞여도 표 열이 맞도록)."""
    space = " " * max(0, width - display_width(text))
    return space + text if right else text + space


def print_section(title):
    print()
    print("#" + "-" * 40)
    print("# " + title)
    print("#" + "-" * 40)


def format_score(score):
    """점수를 repr로 출력해 부동소수점 오차(예: 0.8999999999999999)를 숨기지 않는다."""
    return repr(score)


def format_epsilon():
    """EPSILON을 1e-9 형태로 표시한다 (기본 형식은 1e-09)."""
    return "{:g}".format(EPSILON).replace("e-0", "e-")


def format_value(value):
    """행렬 값 표시용: 정수 값은 0, 1처럼, 아니면 그대로 보여준다."""
    return str(int(value)) if value.is_integer() else repr(value)


def print_matrix(matrix, indent="  "):
    for row in matrix.cells:
        print(indent + " ".join(format_value(v) for v in row))


# ============================================================
# 모드 1: 사용자 입력 (3×3)
# ============================================================

USER_SIZE = 3


def run_user_mode(input_fn=None):
    n = USER_SIZE

    print_section("[1] 필터 입력")
    filter_a = read_matrix(n, "필터 A", input_fn)
    print()
    filter_b = read_matrix(n, "필터 B", input_fn)

    print_section("[2] 저장 확인")
    print("필터 A ({0}×{0})".format(n))
    print_matrix(filter_a)
    print("필터 B ({0}×{0})".format(n))
    print_matrix(filter_b)

    print_section("[3] 패턴 입력")
    pattern = read_matrix(n, "패턴", input_fn)

    # 연산: 점수 계산과 시간 측정 (출력은 측정 구간 밖)
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    pair_ms = measure_average_ms(lambda: (mac(pattern, filter_a), mac(pattern, filter_b)))
    single_ms = measure_average_ms(lambda: mac(pattern, filter_a))
    decision = judge(score_a, score_b, "A", "B")

    print_section("[4] MAC 결과")
    print("A 점수: {}".format(format_score(score_a)))
    print("B 점수: {}".format(format_score(score_b)))
    print("연산 시간(A·B 두 필터, 평균/{}회): {:.6f} ms".format(REPEAT, pair_ms))
    if decision == UNDECIDED:
        print("판정: 판정 불가 (|A-B| < {})".format(format_epsilon()))
    else:
        print("판정: {}".format(decision))

    print_section("[5] 성능 분석 (MAC 1회, 평균/{}회)".format(REPEAT))
    print_performance_table([(n, single_ms)])
    return decision


# ============================================================
# 모드 2: data.json 분석
# ============================================================

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

FILTER_KEY_RE = re.compile(r"^size_(\d+)$")        # filters 키: size_{N}
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")  # patterns 키: size_{N}_{idx}


def load_json(path):
    """JSON 파일을 읽어 (data, 오류 메시지)를 돌려준다. 실패해도 예외를 밖으로 내지 않는다."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, "파일을 찾을 수 없습니다: {}".format(path)
    except json.JSONDecodeError as error:
        return None, "JSON 문법 오류 ({}행 {}열): {}".format(error.lineno, error.colno, error.msg)
    except (OSError, UnicodeDecodeError) as error:
        return None, "파일을 읽을 수 없습니다: {}".format(error)
    if not isinstance(data, dict):
        return None, "최상위 구조가 객체({{...}})가 아닙니다: {}".format(type(data).__name__)
    return data, None


def load_filters(section):
    """filters 섹션을 검증해 ({N: {Cross: Matrix, X: Matrix}}, 로그 목록)을 돌려준다.

    로그는 (성공 여부, 키, 메시지) 튜플이다. 문제가 있는 크기는 결과에서 빠지고,
    그 크기의 패턴은 이후 케이스 단위로 FAIL 처리된다.
    """
    filters = {}
    logs = []
    if not isinstance(section, dict):
        logs.append((False, "filters", "filters 항목이 없거나 객체가 아닙니다"))
        return filters, logs

    for key, pair in section.items():
        match = FILTER_KEY_RE.match(key)
        if not match:
            logs.append((False, key, "키 형식 오류 (size_N 형태가 아님)"))
            continue
        n = int(match.group(1))
        if not isinstance(pair, dict):
            logs.append((False, key, "필터 묶음이 객체가 아닙니다"))
            continue

        loaded = {}
        problems = []
        for raw_label, rows in pair.items():
            label = normalize_label(raw_label)
            if label is None:
                problems.append("알 수 없는 필터 키 '{}'".format(raw_label))
                continue
            if label in loaded:
                problems.append("'{}' 필터가 중복됨 ('{}')".format(label, raw_label))
                continue
            try:
                matrix = Matrix.from_rows(rows)
            except ValueError as error:
                problems.append("{} 필터 오류: {}".format(label, error))
                continue
            if matrix.n != n:
                problems.append(
                    "{} 필터 크기 불일치: 키는 {n}×{n}인데 데이터는 {m}×{m}".format(label, n=n, m=matrix.n)
                )
                continue
            loaded[label] = matrix

        for label in (CROSS, X):
            if label not in loaded and not any(p.startswith(label + " ") for p in problems):
                problems.append("{} 필터 없음".format(label))
        if problems:
            logs.append((False, key, "; ".join(problems)))
        else:
            filters[n] = loaded
            logs.append((True, key, "필터 로드 완료 (Cross, X)"))
    return filters, logs


# FAIL 사유 범주 (README의 실패 원인 분석과 대응)
REASON_SCHEMA = "[스키마]"   # 데이터/스키마 문제: 키 형식, 필드 누락, 필터 없음, input 형식
REASON_SIZE = "[크기]"       # 데이터/스키마 문제: 패턴과 필터 크기 불일치
REASON_LABEL = "[라벨]"      # 데이터/스키마 문제: expected 라벨 해석 불가
REASON_TIE = "[동점]"        # 수치 비교 문제: |Cross-X| < EPSILON
REASON_MISMATCH = "[판정]"   # 로직(또는 데이터) 문제: 판정과 expected가 다름
REASON_ERROR = "[예외]"      # 예상하지 못한 오류


class CaseResult:
    """패턴 케이스 하나의 분석 결과."""

    def __init__(self, key):
        self.key = key
        self.score_cross = None
        self.score_x = None
        self.decision = None
        self.expected = None
        self.passed = False
        self.reason = None

    def fail(self, category, message):
        self.passed = False
        self.reason = "{} {}".format(category, message)
        return self


def evaluate_case(key, entry, filters):
    """패턴 하나를 검증·연산·판정한다. 문제가 있으면 예외 대신 FAIL 결과를 돌려준다."""
    result = CaseResult(key)

    match = PATTERN_KEY_RE.match(key)
    if not match:
        return result.fail(REASON_SCHEMA, "키 형식 오류: size_N_idx 형태가 아닙니다")
    n = int(match.group(1))

    if not isinstance(entry, dict):
        return result.fail(REASON_SCHEMA, "케이스 값이 객체가 아닙니다")
    missing = [field for field in ("input", "expected") if field not in entry]
    if missing:
        return result.fail(REASON_SCHEMA, "필드 누락: {}".format(", ".join(missing)))

    raw_expected = entry["expected"]
    result.expected = normalize_label(raw_expected)

    if n not in filters:
        return result.fail(REASON_SCHEMA, "size_{} 필터가 없거나 로드에 실패했습니다".format(n))

    try:
        pattern = Matrix.from_rows(entry["input"])
    except ValueError as error:
        return result.fail(REASON_SCHEMA, "input 오류: {}".format(error))
    if pattern.n != n:
        return result.fail(
            REASON_SIZE,
            "키는 {n}×{n}인데 input은 {m}×{m} (필터 {n}×{n}와 크기 불일치)".format(n=n, m=pattern.n),
        )

    result.score_cross = mac(pattern, filters[n][CROSS])
    result.score_x = mac(pattern, filters[n][X])
    result.decision = judge(result.score_cross, result.score_x, CROSS, X)

    if result.expected is None:
        return result.fail(REASON_LABEL, "expected 값 {!r}을(를) 해석할 수 없습니다".format(raw_expected))
    if result.decision == UNDECIDED:
        return result.fail(
            REASON_TIE,
            "|Cross-X| = {:.3g} < {} → UNDECIDED 규칙에 따라 FAIL".format(
                abs(result.score_cross - result.score_x), format_epsilon()
            ),
        )
    if result.decision != result.expected:
        return result.fail(
            REASON_MISMATCH, "판정 {} ≠ expected {}".format(result.decision, result.expected)
        )
    result.passed = True
    return result


def print_case(result):
    print("--- {} ---".format(result.key))
    if result.score_cross is not None:
        print("Cross 점수: {}".format(format_score(result.score_cross)))
        print("X 점수: {}".format(format_score(result.score_x)))
    print("판정: {} | expected: {} | {}".format(
        result.decision or "-",
        result.expected or "-",
        "PASS" if result.passed else "FAIL",
    ))
    if result.reason:
        print("  사유: {}".format(result.reason))


def analyze_patterns(section, filters):
    """patterns 섹션의 모든 케이스를 판정한다. 한 케이스의 오류가 전체를 멈추지 않는다."""
    results = []
    if not isinstance(section, dict):
        print("patterns 항목이 없거나 객체가 아닙니다. 분석할 케이스가 없습니다.")
        return results
    for key, entry in section.items():
        try:
            result = evaluate_case(key, entry, filters)
        except Exception as error:  # 예상하지 못한 오류도 케이스 단위 FAIL로 처리
            result = CaseResult(key).fail(
                REASON_ERROR, "{}: {}".format(type(error).__name__, error)
            )
        print_case(result)
        results.append(result)
    return results


def summarize(results):
    """결과 목록에서 (총, 통과, 실패, 사유 범주별 건수)를 계산한다."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    categories = {}
    for r in results:
        if not r.passed:
            category = r.reason.split(" ", 1)[0] if r.reason else REASON_ERROR
            categories[category] = categories.get(category, 0) + 1
    return total, passed, total - passed, categories


def print_summary(results):
    total, passed, failed, categories = summarize(results)
    print("총 테스트: {}개".format(total))
    print("통과: {}개".format(passed))
    print("실패: {}개".format(failed))
    if not failed:
        print()
        print("실패 케이스 없음")
        return
    print()
    print("실패 원인 분류: " + ", ".join(
        "{} {}건".format(category, count) for category, count in categories.items()
    ))
    print()
    print("실패 케이스:")
    for r in results:
        if not r.passed:
            print("- {}: {}".format(r.key, r.reason))


def run_json_mode(path=DEFAULT_DATA_PATH):
    """data.json 분석: 필터 로드 → 패턴 분석. 결과 목록을 돌려주고, 파일을 못 읽으면 None."""
    print("데이터 파일: {}".format(path))
    data, error = load_json(path)
    if error:
        print("✗ 데이터를 불러오지 못했습니다: {}".format(error))
        return None

    print_section("[1] 필터 로드")
    filters, logs = load_filters(data.get("filters"))
    for ok, key, message in logs:
        print("{} {} {}".format("✓" if ok else "✗", pad(key, 8), message))

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = analyze_patterns(data.get("patterns"), filters)

    print_section("[3] 성능 분석 (MAC 1회, 평균/{}회)".format(REPEAT))
    comparison = measure_comparison(PERF_SIZES, filters)
    print_performance_table([(n, ms_2d) for n, ms_2d, _, _ in comparison])
    print()
    print("[보너스] 메모리 접근 최적화: 2차원 → 1차원 배열")
    print_comparison_table(comparison)

    print_section("[4] 결과 요약")
    print_summary(results)
    return results


# ============================================================
# 메뉴 / 진입점
# ============================================================

def choose_mode(input_fn=None):
    """모드를 선택받는다. 1 또는 2가 아니면 다시 입력받는다."""
    input_fn = input_fn or input
    print()
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    while True:
        choice = input_fn("선택: ").strip()
        if choice in ("1", "2"):
            return choice
        print("  1 또는 2를 입력하세요.")


def configure_output():
    """Windows 콘솔·파이프에서도 한글이 깨지지 않도록 출력 인코딩을 UTF-8로 맞춘다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    """사용법: python main.py [데이터 파일 경로]  (기본값: main.py 옆의 data.json)"""
    configure_output()
    argv = sys.argv[1:] if argv is None else argv
    data_path = argv[0] if argv else DEFAULT_DATA_PATH

    print("=== Mini NPU Simulator ===")
    try:
        if choose_mode() == "1":
            run_user_mode()
        elif run_json_mode(data_path) is None:
            return 1
    except (EOFError, KeyboardInterrupt):
        print()
        print("입력이 종료되어 프로그램을 마칩니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
