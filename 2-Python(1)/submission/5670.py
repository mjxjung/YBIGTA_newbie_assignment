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

            if self[index_at].children[position] is None:
                # Create new node and link it
                self.append(TrieNode(body=i))
                self[index_at].children[position] = len(self) - 1

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
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        # new_index = None # 구현하세요!
        # 다음 인덱스 탐색
        position = ord(element) - ord('a')  # 현재 문자의 children 배열에서의 위치
        new_index = trie[pointer].children[position]  # 해당 위치의 자식 노드 인덱스

        if new_index is None:
            break  # 다음 노드가 없으면 종료

        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    # 구현하세요!
    """
    function main, computing the average # of button presses necessary to insert every word in a dictionary.
    """
    # input data handling
    input_data = sys.stdin.read().strip().split('\n')
    num_word = int(input_data[0])  # 단어의 개수
    words = input_data[1:]

    # make instance of Trie
    trie: Trie[str] = Trie() # using type annotation

    for word in words:
        trie.push(word)

    # 버튼 입력 횟수 계산
    total_presses = sum(count(trie, word) for word in words)

    # 평균 계산 및 출력
    print(f"{total_presses / num_word:.2f}")


if __name__ == "__main__":
    main()