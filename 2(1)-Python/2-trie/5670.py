from lib import Trie
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