"""Mini NPU Simulator.

MAC(Multiply-Accumulate) 연산으로 N×N 패턴이 Cross(십자가)인지 X인지 판별한다.
외부 라이브러리 없이 Python 표준 라이브러리만 사용한다.
"""

import math
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


def read_matrix(n, title, input_fn=input):
    """n줄을 한 줄씩 입력받아 n×n Matrix를 만든다.

    잘못된 줄은 안내 문구를 보여주고 그 줄만 다시 입력받는다.
    input_fn은 테스트에서 입력을 흉내 낼 수 있도록 바꿔 끼울 수 있다.
    """
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


def print_performance_table(rows):
    """rows: (n, 평균 ms) 목록. 크기 / 평균 시간 / 연산 횟수(N²) 표를 출력한다."""
    print(pad("크기", 10) + pad("평균 시간(ms)", 16, right=True) + pad("연산 횟수(N²)", 16, right=True))
    print("-" * 42)
    for n, avg_ms in rows:
        print(
            pad("{0}×{0}".format(n), 10)
            + pad("{:.6f}".format(avg_ms), 16, right=True)
            + pad(str(n * n), 16, right=True)
        )


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


def run_user_mode(input_fn=input):
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
        print("판정: 판정 불가 (|A-B| < {:g})".format(EPSILON))
    else:
        print("판정: {}".format(decision))

    print_section("[5] 성능 분석 (MAC 1회, 평균/{}회)".format(REPEAT))
    print_performance_table([(n, single_ms)])
    return decision


# ============================================================
# 모드 2: data.json 분석
# ============================================================

def run_json_mode():
    print("data.json 분석 모드는 아직 구현 중입니다. (4단계)")


# ============================================================
# 메뉴 / 진입점
# ============================================================

def choose_mode(input_fn=input):
    """모드를 선택받는다. 1 또는 2가 아니면 다시 입력받는다."""
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


def main():
    configure_output()
    print("=== Mini NPU Simulator ===")
    try:
        if choose_mode() == "1":
            run_user_mode()
        else:
            run_json_mode()
    except (EOFError, KeyboardInterrupt):
        print()
        print("입력이 종료되어 프로그램을 마칩니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
