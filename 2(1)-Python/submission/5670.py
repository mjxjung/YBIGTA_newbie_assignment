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
        for element in seq:
            # 현재 노드의 children에서 element를 가진 노드 탐색
            child_index = self._find_or_create_child(index_at, element)
            index_at = child_index  # 다음 노드로 이동

        # 마지막 노드를 단어의 끝으로 표시
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
            child_index = self._find_child(current_index, element)
            if child_index is None:
                return False  # 현재 prefix가 존재하지 않음
            current_index = child_index

        return True  # Prefix exists

    def _find_or_create_child(self, parent_index: int, element: T) -> int:
        """
        Find the child node with the given element or create a new one if it doesn't exist.

        Args:
            parent_index (int): The index of the parent node.
            element (T): The element to find or insert.

        Returns:
            int: The index of the found or newly created child node.
        """
        # 부모 노드의 자식 노드 리스트에서 일치하는 body를 가진 노드 탐색
        for child_index in self[parent_index].children:
            if self[child_index].body == element:
                return child_index  # 이미 존재하는 자식 노드 반환

        # 존재하지 않으면 새로운 노드를 생성하고 추가
        new_node = TrieNode(body=element)
        self.append(new_node)
        new_index = len(self) - 1
        self[parent_index].children.append(new_index)
        return new_index

    def _find_child(self, parent_index: int, element: T) -> Optional[int]:
        """
        Find the child node with the given element.

        Args:
            parent_index (int): The index of the parent node.
            element (T): The element to find.

        Returns:
            Optional[int]: The index of the found child node, or None if it doesn't exist.
        """
        # 부모 노드의 자식 노드 리스트에서 일치하는 body를 가진 노드 탐색
        for child_index in self[parent_index].children:
            if self[child_index].body == element:
                return child_index
        return None  # 일치하는 자식 노드가 없으면 None 반환


trie = Trie[str]()
trie.push("IVO")
trie.push("JASNA")
trie.push("JOSIPA")

print(trie.count_prefix("IVO"))  # True 예상
print(trie.count_prefix("JA"))   # True 예상
print(trie.count_prefix("JO"))   # True 예상



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