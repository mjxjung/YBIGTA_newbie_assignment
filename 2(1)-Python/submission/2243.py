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


def main() -> None:
    # 구현하세요!
    """
    Main function to process candy box operations.
    """
    input = sys.stdin.read
    data = input().strip().split('\n')

    n = int(data[0])  # Number of operations
    operations = data[1:]

    MAX_TASTE = 1_000_000

    # Initialize SegmentTree for 1 to MAX_TASTE
    seg_tree: SegmentTree[int, int] = SegmentTree(MAX_TASTE, func=lambda x, y: x + y, default=0)


    results = []

    for operation in operations:
        args = list(map(int, operation.split()))
        if args[0] == 1:  # Remove candy
            k = args[1]
            taste = seg_tree.query(k)
            results.append(taste)
            seg_tree.update(taste, -1)
        elif args[0] == 2:  # Add candy
            taste, count = args[1], args[2]
            seg_tree.update(taste, count)

    # Print results for all "1" operations
    sys.stdout.write('\n'.join(map(str, results)) + '\n')


if __name__ == "__main__":
    main()