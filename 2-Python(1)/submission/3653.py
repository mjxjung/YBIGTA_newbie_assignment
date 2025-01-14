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
        Main function to process DVD stack operations.
        """
    num_dvd = sys.stdin.read
    data = num_dvd().strip().split('\n')

    test_cases = int(data[0])  # Number of test cases
    results = []

    index = 1
    for _ in range(test_cases):
        # Parse the test case
        n, m = map(int, data[index].split())
        movies = list(map(int, data[index + 1].split()))
        index += 2

        # SegmentTree for positions
        max_size = n + m
        seg_tree: SegmentTree[int, int] = SegmentTree(max_size, func=lambda x, y: x + y, default=0)

        # Initialize positions and SegmentTree
        positions = [0] * (n + 1)
        for i in range(1, n + 1):
            positions[i] = m + i - 1  # Initial position in the stack
            seg_tree.update(positions[i], 1)

        # Process each movie access
        current_top = m - 1
        test_result = []
        for movie in movies:
            pos = positions[movie]
            # Count DVDs above the current movie
            count_above = seg_tree.query(pos, pos)
            test_result.append(count_above)

            # Update SegmentTree and move the movie to the top
            seg_tree.update(pos, -1)
            current_top -= 1
            positions[movie] = current_top
            seg_tree.update(current_top, 1)

        results.append(" ".join(map(str, test_result)))

    # Print results for all test cases
    sys.stdout.write("\n".join(results) + "\n")


if __name__ == "__main__":
    main()