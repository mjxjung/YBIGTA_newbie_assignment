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