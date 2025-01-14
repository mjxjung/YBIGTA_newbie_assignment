from lib import SegmentTree
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