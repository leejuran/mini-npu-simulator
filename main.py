"""Mini NPU Simulator.

MAC(Multiply-Accumulate) 연산으로 N×N 패턴이 Cross(십자가)인지 X인지 판별한다.
외부 라이브러리 없이 Python 표준 라이브러리만 사용한다.
"""

import math


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
