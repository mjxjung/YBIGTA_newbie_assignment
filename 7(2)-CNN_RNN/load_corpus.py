# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    """
    여러 NLP 데이터셋을 로드하여 모든 텍스트 데이터를 하나의 리스트로 수집하는 함수.

    Returns:
        corpus (list[str]): 수집된 텍스트 데이터 리스트
    """
    corpus: list[str] = []
    # 구현하세요!
    # Poem Sentiment 데이터셋 추가
    poem_dataset = load_dataset("google-research-datasets/poem_sentiment")
    for split in ("train", "validation", "test"):
        for example in poem_dataset[split]:
            corpus.append(example["verse_text"])  # 시 데이터 추가

    return corpus