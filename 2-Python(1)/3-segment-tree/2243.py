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