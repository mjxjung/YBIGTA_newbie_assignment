from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    # 구현하세요!
    def __init__(self, size: int, func: Callable[[T, T], T], default: T) -> None:
        """
        Initialize the segment tree.

        Args:
            - size (int): The size of the array for which the segment tree is built.
            - func (Callable[[T, T], T]): The function used for range queries (e.g., min, max, sum).
            - default (T): The default value to fill unused nodes (e.g., 0 for sum, inf for min).
        """
        self.n = size
        self.func = func
        self.default = default
        self.tree = [default] * (4 * self.n)

    def update(self, index: int, diff: T, node: int = 0, start: int = 0, end: Optional[int] = None) -> None:
        """
        Update a value in the segment tree.

        Args:
            - index (int): The index to update.
            - diff (T): The value to add (can be positive or negative).
            - node (int): Current tree node index.
            - start (int): Start index of the segment.
            - end (Optional[int]): End index of the segment.
        """
        if end is None:
            end = self.n - 1

        if start == end:
            self.tree[node] = self.func(self.tree[node], diff)
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            if start <= index <= mid:
                self.update(index, diff, left_child, start, mid)
            else:
                self.update(index, diff, right_child, mid + 1, end)

            self.tree[node] = self.func(self.tree[left_child], self.tree[right_child])

    def query(self, k: int, node: int = 0, start: int = 0, end: Optional[int] = None) -> T:
        """
        Find the k-th element in the segment tree.

        Args:
            - k (int): The k-th smallest element to find.
            - node (int): Current tree node index.
            - start (int): Start index of the segment.
            - end (Optional[int]): End index of the segment.

        Returns:
            - int: The index of the k-th element.
        """
        if end is None:
            end = self.n - 1

        if start == end:
            return start

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        left_value = self.tree[left_child]

        if isinstance(left_value, int) and left_value >= k:
            return self.query(k, left_child, start, mid)
        elif isinstance(k, int) and isinstance(left_value, int):
            return self.query(k - left_value, right_child, mid + 1, end)

        raise ValueError("Invalid types for k or tree values in query.")



import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # 구현하세요!
    """
    Main function to process range maximum sum queries and updates.
    """
    input = sys.stdin.read
    data = input().strip().split('\n')

    n = int(data[0])  # Size of the array
    arr = list(map(int, data[1].split()))
    m = int(data[2])  # Number of queries
    queries = data[3:]

    # Initialize segment tree
    seg_tree: SegmentTree[Pair, None] = SegmentTree(n, func=Pair.f_merge, default=Pair.default())

    # Initialize the segment tree using updates
    for i, value in enumerate(arr):
        seg_tree.update(i, Pair.f_conv(value))

    results = []

    for query in queries:
        parts = query.split()
        if parts[0] == '1':  # Update query
            i, v = int(parts[1]), int(parts[2])
            seg_tree.update(i - 1, Pair.f_conv(v))
        elif parts[0] == '2':  # Range maximum sum query
            l, r = int(parts[1]), int(parts[2])
            # Ensure the query returns a Pair and call sum()
            pair_result = seg_tree.query(l - 1, r - 1)
            results.append(pair_result.sum())

    # Output results for range queries
    sys.stdout.write('\n'.join(map(str, results)) + '\n')


if __name__ == "__main__":
    main()