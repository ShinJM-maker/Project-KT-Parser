## KT-Parser

본 저장소는 PNU AILAB KT-Parser 과제의 Dependency Parsing model 코드 공유 및 성과를 공유하기 위한 것입니다.

## 모델구조
https://user-images.githubusercontent.com/66815358/182843232-7f01c746-0804-4b95-91c3-895886b71727.png

## Dependencies

requirements.txt에 해당하는 패키지들을 설치하여 개발 환경을 구축해야 합니다.

Make sure you have installed the packages listed in requirements.txt.

다음과 같은 명령어를 통해서 필요한 패키지를 설치할 수 있습니다.

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

모든 실험들은 파이썬 3.7 버전에서 진행되었으며, 패키지들의 Dependency를 위해서 3.7 버전(혹은 이상)의 파이썬 사용을 권장드립니다.

All expereiments are tested under Python 3.7 environment.


## 모델 구현 코드

DP Parser의 Torch 코드는 /baseline/models/dependency_parsing.py에 작성이 되어있습니다. (DPTransformer 클래스)

## 데이터셋

데이터셋의 용량이 큰 관계로 아래의 공유링크를 사용하여 다운로드 받아주시기 바랍니다. 받은 데이터셋은 data/klue_benchmark에 다운로드받은 폴더를 그대로 넣어주시면 됩니다.
http://pnuailab.synology.me/sharing/ivl3ZfN6p


## 실행

파서 코드 실행을 위해 필요한 변수는 다음과 같습니다.

OUTPUT_DIR="klue_output"

DATA_DIR="data/klue_benchmark"

VERSION="v1.1"

task="dp"

OUTPUT_DIR은 학습된 모델 및 실험 결과가 저장될 위치를 나타내며, DATA_DIR은 학습 및 평가를 위한 데이터의 위치를 나타냅니다.

학습 및 평가를 위해서는 다음의 명령어를 실행해주시면 됩니다.

주요 파라미터

model_name_or_path: 학습에 사용될 Huggingface에 배포된 모델의 URL을 URL을 넣어주시면 됩니다. 현재는 monologg/koelectra-base-v3-discriminator를 사용하고 있습니다.

learning_rate: 모델의 학습률을 지정합니다.

train_batch_size: 학습을 위한 배치 사이즈를 나타내며, 사용하시는 GPU의 메모리에 맞게 설정하시면 됩니다. 현재는 RTX-3090(24Gb)에 맞게 설정하였습니다.

warmup_ratio: 지정한 비율만큼 처음 학습을 시작했을 때, 훨씬 작은 Learning Rate로 학습을 하도록 하는 파라미터입니다.


- 학습 명령어

python dp_main.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 4


- 평가 명령어

python dp_main.py evaluate --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 36



- 테스트 명령어

python dp_main.py test --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 36


## 학습된 모형 배포(4월 버전)

http://pnuailab.synology.me/sharing/irbmcpyIY


## 데모 실행 버전(Inference 예정)

추후 업데이트 할 예정입니다(8월 중 예상)

## Reference

https://www.koreascience.or.kr/article/CFKO202130060562801.pdf

https://github.com/KLUE-benchmark/KLUE


## Announce

Inference : 규칙 적용 Layer로 시스템 구조가 복잡해져 튜닝과정에 있으며 8월 중 업로드할 예정입니다.
