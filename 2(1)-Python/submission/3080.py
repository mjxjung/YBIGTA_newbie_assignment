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

    # 입력 데이터 읽기
    name_input = sys.stdin.read().strip().split("\n")
    print("Input Data:", name_input)  # 입력 데이터 확인
    num_word = int(name_input[0])
    name_seq = name_input[1:]

    # Trie 생성 및 이름 추가
    trie = Trie[str]()
    for name in sorted(name_seq):
        trie.push(name)

    # DP 배열 초기화
    dp = [0] * (num_word + 1)
    dp[0] = 1

    # DP 계산
    for i in range(1, num_word + 1):
        current_name = name_seq[i - 1]
        for j in range(i):
            prefix_name = name_seq[j]
            # Trie에서 접두사를 확인하고 current_name이 prefix_name으로 시작하는지 확인
            if trie.count_prefix(prefix_name) and current_name.startswith(prefix_name):
                dp[i] = (dp[i] + dp[j]) % mod
        print(f"dp[{i}] = {dp[i]}")  # DP 배열 값 출력

    # 최종 결과 출력
    print(dp[num_word])

if __name__ == "__main__":
    main()