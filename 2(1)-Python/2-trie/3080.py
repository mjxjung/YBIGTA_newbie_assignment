from lib import Trie
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