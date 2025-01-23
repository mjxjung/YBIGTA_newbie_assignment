from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        inserting and saving a sequence into the Trie.

        Args:
            seq (Iterable[T]): sequence to be inserted, list or int or whatever...
        """
        # 구현하세요!
        index_at = 0
        for i in seq:
            position = self._get_position(i)

            # Ensure children list is large enough to accommodate position
            while len(self[index_at].children) <= position:
                self[index_at].children.append(None)

            # Create new node if needed
            if self[index_at].children[position] is None:
                self.append(TrieNode(body=i))
                self[index_at].children[position] = len(self) - 1

            # Move to the child node
            index_at = self[index_at].children[position]

        # Mark the end of the sequence
        self[index_at].is_end = True

    # 구현하세요!
    def count_prefix(self, prefix: Iterable[T]) -> bool:
        """
        Check if a given prefix exists in the Trie.

        Args:
            - prefix (Iterable[T]): The prefix to check.

        Returns:
            - bool: True if the prefix exists, False otherwise.
        """
        current_index = 0
        for element in prefix:
            position = self._get_position(element)

            # Ensure children list is large enough to accommodate position
            if position >= len(self[current_index].children):
                return False

            # Check if the child node exists
            if self[current_index].children[position] is None:
                return False

            current_index = self[current_index].children[position]

        return True  # Prefix exists

    def _get_position(self, element: T) -> int:
        """
        Helper method to get the position/index for a given element.
        This method assumes that the elements are characters (e.g., a-z).

        Args:
        - element (T): The element to determine its position.

        Returns:
        - int: The index position in the children array.
        """
        if isinstance(element, str):
            return ord(element) - ord('a')  # Assuming input is lowercase alphabets
        raise ValueError("Unsupported element type for position computation")


import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    """
    function main, calculating the # of correct arrangements for name sequences

    Input:
        - First line: Integer N (number of names, 3 ≤ N ≤ 3000)
        - Next N lines: Names consisting of uppercase alphabets (length < 3000, unique)

    Output:
        - An integer representing the number of valid arrangements modulo 1,000,000,007.
        """
    mod = 1_000_000_007

    # input data handling
    name_input = sys.stdin.read().strip().split('\n')
    num_word = int(name_input[0])
    name_seq = name_input[1:]

    # make instance of Trie
    trie: Trie[str] = Trie() # using type annotation

    for n in sorted(name_seq):  # 이름을 a to z로 정렬
        trie.push(n.lower())  # 이름을 Trie에 삽입 (소문자로 통일)

    # DP 테이블 초기화
    dp = [0] * (num_word + 1)
    dp[0] = 1

    # DP 계산
    for i in range(1, num_word + 1):
        front = name_seq[i - 1].lower()
        for j in range(i):
            if trie.count_prefix(front[:len(name_seq[j])].lower()):
                dp[i] = (dp[i] + dp[j]) % mod

    # 결과 출력
    print(dp[num_word])


if __name__ == "__main__":
    main()