## KT-Parser

본 저장소는 PNU AILAB KT-Parser 과제의 Dependency Parsing model 코드 공유 및 성과를 공유하기 위한 것입니다.

## 의존 구문 분석이란
- 자연어 문장을 지배소-피지배소 의존 관계로 분석하는 구문 분석 방법론

- 문장의 구조적 중의성 해소 가능

- 어순이 고정적이지 않고 문장 성분의 생략이 빈번한 한국어에 적합

- 레이블 : 구문태그_기능태그 형태로 태그를 결합하여 사용
    예) NP_SBJ, VP_MOD

- 평가 Metric: UAS(지배소 정답 개수), LAS(레이블 정답 개수) 

![image](https://user-images.githubusercontent.com/66815358/182911626-3c366a7d-1dd1-493a-b023-d86d37569114.png)



## 모델구조
![모델구조](https://user-images.githubusercontent.com/66815358/182843400-f191a3f7-6b99-4c42-9424-07b2ae4b06ec.png)

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
- 21세기 세종계획 구구조 구문분석 말뭉치를 포함한 200만 어절(14만 5천 문장)
    - 학습(Train) 12만 문장, 개발(Dev) 1만 5천 문장, 평가(Test) 1만 문장
- 가이드라인 : 정보통신단체표준(TTAS)
    - 의존 구문분석 말뭉치 구축을 위한 의존관계 태그세트 및 의존관계 설정방법(2015, TTAS 표준)
- 태깅 에러
    - 추정 품질 평균 : 93.57% -> 추정 태깅 에러 6.43%(지배소에러 + 레이블에러)
        - 구문 및 무형 대용어 복원 말뭉치 연구 분석(2021, 국립국어원)
    - 데이터셋의 태깅에러 -> 모델의 학습과 예측에 오류 발생



- 데이터셋의 용량이 큰 관계로 아래의 공유링크를 사용하여 다운로드 받아주시기 바랍니다. 받은 데이터셋은 data/klue_benchmark에 다운로드받은 폴더를 그대로 넣어주시면 됩니다.
http://pnuailab.synology.me/sharing/ivl3ZfN6p

## Metrics
UAS
LAS


## 실행

파서 코드 실행을 위해 필요한 변수는 다음과 같습니다.

OUTPUT_DIR="klue_output"

DATA_DIR="data/klue_benchmark"

VERSION="v1.1"

task="dp"

OUTPUT_DIR은 학습된 모델 및 실험 결과가 저장될 위치를 나타내며, DATA_DIR은 학습 및 평가를 위한 데이터의 위치를 나타냅니다.

학습 및 평가를 위해서는 다음의 명령어를 실행해주시면 됩니다.

- 주요 파라미터

model_name_or_path: 학습에 사용될 Huggingface에 배포된 모델의 URL을 URL을 넣어주시면 됩니다. 현재는 monologg/koelectra-base-v3-discriminator를 사용하고 있습니다.

learning_rate: 모델의 학습률을 지정합니다.

train_batch_size: 학습을 위한 배치 사이즈를 나타내며, 사용하시는 GPU의 메모리에 맞게 설정하시면 됩니다. 현재는 RTX-3090(24Gb)에 맞게 설정하였습니다.

warmup_ratio: 지정한 비율만큼 처음 학습을 시작했을 때, 훨씬 작은 Learning Rate로 학습을 하도록 하는 파라미터입니다.


- 학습 명령어

python dp_main.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 4


- 평가 명령어

python dp_main.py evaluate --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 3 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 36



- 테스트 명령어

python dp_main.py test --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 3 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 36

- 결과 확인
학습 후 model_output 폴더의 Metrics.csv에 매 validation step마다 저장이 되어있습니다. 왼쪽부터 Micro UAS, Macro UAS, Micro LAS, Macro LAS 이며 정확한 결과는 두번째 네번째인 Macro UAS와  Macro LAS를 확인하시면 됩니다.

## 학습 결과
UAS 95.76 LAS 92.55

## 결과 분석
- 성능 향상
    - 과제 시작(1월 대비) 7월 보고 기준 UAS 5.52, LAS 7.42% 향상
![성능결과(8-05)](https://user-images.githubusercontent.com/66815358/182917840-74f95216-ac07-4523-86f1-661276c9be6d.png)

- 타 모델들과 성능 비교
    - 

## 모델 특징
- 데이터에 태깅 에러가 있음에도 잘 예측할 수 있다 : 오류가 있어도 됨

- 학습시간을 줄일 수 있다 : 

- 설명이 가능하다 : 

- 다른모델에도 적용할 수 있다(일반화) : 

## 

## 학습된 모형 배포

http://pnuailab.synology.me/sharing/irbmcpyIY


## 데모 실행 버전(Inference 예정)

추후 업데이트 할 예정입니다(8월 중 예상)

## Reference

https://www.koreascience.or.kr/article/CFKO202130060562801.pdf

https://github.com/KLUE-benchmark/KLUE


## Announce

Inference : 규칙 적용 Layer로 전처리 과정과 모델 파라미터가 복잡해져 추가 튜닝과정에 있으며, 8월 중 업로드할 예정입니다.
