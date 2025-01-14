#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
## TODO
if ! command -v conda &> /dev/null; then
    echo "Miniconda가 설치되어 있지 않습니다. 설치를 진행합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    echo "Miniconda 설치 완료."
else
    echo "Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
## TODO
ENV_NAME="myenv"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "가상환경 $ENV_NAME 생성 중..."
    conda create -n "$ENV_NAME" python=3.9 -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    filename=$(basename "$file" .py)  # 파일명 추출 (확장자 제거)
    input_file="../input/${filename}_input"  # 입력 파일 경로
    output_file="../output/${filename}_output"  # 출력 파일 경로

    echo "실행 중: $file"
    if [ -f "$input_file" ]; then
        python "$file" < "$input_file" > "$output_file"
        echo "$file 실행 완료. 결과는 $output_file에 저장되었습니다."
    else
        echo "입력 파일 $input_file이 존재하지 않습니다. $file을 건너뜁니다."
    fi
done

# mypy 테스트 실행행
## TODO
echo "mypy 타입 검사 실행 중..."
for file in *.py; do
    mypy "$file"
done

# 가상환경 비활성화
## TODO
conda deactivate