## KT-Parser

본 저장소는 PNU AILAB KT-Parser 과제의 Dependency Parsing model 코드 공유 및 성과를 공유하기 위한 것이다.

## 의존 구문 분석이란
- 자연어 문장을 지배소-피지배소 의존 관계로 분석하는 구문 분석 방법론

- 문장의 구조적 중의성 해소 가능

- 어순이 고정적이지 않고 문장 성분의 생략이 빈번한 한국어에 적합

- 의존관계 레이블 : 구문태그_기능태그 형태로 태그를 결합하여 사용
    예) NP_SBJ, VP_MOD

- 평가 Metric
    - UAS : 지배소 정답 개수, 해당 어절(의존소)이 어느 어절(지배소)에 의존하는가
    - LAS : 의존관계 레이블 정답 개수, 해당 어절(의존소)이 어느 어절(지배소)과 어떠한 관계를 가지는가

![image](https://user-images.githubusercontent.com/66815358/182911626-3c366a7d-1dd1-493a-b023-d86d37569114.png)



## 모델구조
![모델구조](https://user-images.githubusercontent.com/66815358/182843400-f191a3f7-6b99-4c42-9424-07b2ae4b06ec.png)
- 어절 표상
    - 각 입력 문장은 사전 학습 언어모델을 통해 토큰 단위로 임베딩되어 벡터로 표현된다. 언어모델은 KoELECTRA을 사용하였고, 입력 문장을 KoELECTRA 토크나이저로 wordpiece 단위로 분절 한 뒤 언어 모델에 입력하여 출력된 벡터들의 평균으로 어절 임베딩 벡터를 구성하며, 이러한 벡터에 형태소 품사 임베딩 벡터를 결합하여 어절을 표상한다. 본 모델은 기존 모델이 각 어절의 마지막 형태소 정보만을 사용한 것과 달리, 어절의 마지막과 그 앞 그리고 첫 번째, 총 세 개의 형태소 정보를 사용하여 어절을 표상한다. 어절의 형태소가 3개 미만일 경우엔 NULL값으로 임베딩한다
- 문맥 반영
    - 어절 표상을 순환신경망을 통해 주변 토큰들과의 연관성(문맥)을 반영한다. 기존에 Transformer 인코더를 통해 문맥을 반영했으나 파라미터가 많아짐에 따라 Inference 및 모듈화에 에러가 발생하여 기존 모델들과 같은 Bi-LSTM을 사용하여 문맥을 반영하였다.
- 지배소 및 레이블 인식
    - 문맥 반영된 각 어절 별 임베딩 벡터를 포인터 네트워크를 사용하여 지배소-피지배소 및 의존관계 레이블을 인식한다. 포인터 네트워크는 attention 기법을 통해 각 어절과 다른 모든 어절 간의 상호 의존성을 attention score 형태로써 구한다.
- 규칙기반 Attention 제어(지배소-의존소 제약규칙)
    -  어절은 biaffine문장내 모든 어절들 중에 가장 어텐션 값이 높은 어절을 지배소로 결정하는데, 규칙에 위반되는 경우 attention 값을 0으로 하여 정답 지배소로 가지 못하게 하고, 규칙에 의거한 정답일시 attention 값을 최댓값(100)을 주어 정답 지배소로 가지게 하였다.
    
## 지배소 의존소 제약규칙
- 기존의 딥러닝 기반 모델은 다음과 같은 한계를 가진다
    - 데이터셋에 의존적 : 데이터셋의 정답에 에러가 있거나 혹은 부족할 시에 모델의 예측 결과도 에러가 발생할 수 있음(overfitting)
    - 제어가 불가능 : 명확한 언어학적 규칙을 적용하지 못함
    - EndToEnd 방식 : 결과를 설명할 수 없음
- 딥러닝 기반 모델은 데이터셋에 의존적이기 때문에 데이터셋의 정답에 에러가 있거나 혹은 부족할 시에 모델의 예측 결과도 에러가 발생할 수 있다. 또한 TTA 가이드라인에 따라 의존 구문 분석에는 기본적으로 지켜야할 규칙들이 있으며 데이터셋에 관계없이 적용되어야 하지만 기존 딥러닝 기반 모델은 학습 단계에서 규칙을 적용할 수 없다는 한계가 있다. 추가적으로 현재의 딥러닝 학습 방식은 EndToEnd 방식으로, 한번 학습을 시작하면 출력값만 확인 가능할뿐 왜 그러한 출력이 나오는지 설명 할 수 없다. 본 연구에서는 이를 해결할 방법으로 규칙에 기반한 후처리 Layer를 통해 어텐션을 제약하는 방법인 지배소-의존소 제약규칙을 제안하였다.

## Dependencies

requirements.txt에 해당하는 패키지들을 설치하여 개발 환경을 구축해야 한다.

Make sure you have installed the packages listed in requirements.txt.

다음과 같은 명령어를 통해서 필요한 패키지를 설치할 수 있다.

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

모든 실험들은 파이썬 3.7 버전에서 진행되었으며, 패키지들의 Dependency를 위해서 3.7 버전(혹은 이상)의 파이썬 사용을 권장한다.

All expereiments are tested under Python 3.7 environment.


## 모델 구현 코드

DP Parser의 Torch 코드는 /baseline/models/dependency_parsing.py에 작성이 되어있다. (DPTransformer 클래스)

## 데이터셋
- 21세기 세종계획 구구조 구문분석 말뭉치를 포함한 200만 어절(14만 5천 문장)
    - 학습(Train) 12만 문장, 개발(Dev) 1만 5천 문장, 평가(Test) 1만 문장
- 가이드라인 : 정보통신단체표준(TTAS)
    - 의존 구문분석 말뭉치 구축을 위한 의존관계 태그세트 및 의존관계 설정방법(2015, TTAS 표준)
- 태깅 에러
    - 추정 품질 평균 : 93.57% -> 추정 태깅 에러 6.43%(지배소에러 + 레이블에러)
        - 구문 및 무형 대용어 복원 말뭉치 연구 분석(2021, 국립국어원)
    - 데이터셋의 태깅에러 -> 모델의 학습과 예측에 오류 발생



- 데이터셋의 용량이 큰 관계로 아래의 공유링크를 사용하여 다운로드 받을 수 있다. 받은 데이터셋은 data/klue_benchmark에 다운로드받은 폴더를 그대로 넣으면 된다.
http://pnuailab.synology.me/sharing/ivl3ZfN6p


## 실행

파서 코드 실행을 위해 필요한 변수는 다음과 같다

OUTPUT_DIR="klue_output"

DATA_DIR="data/klue_benchmark"

VERSION="v1.1"

task="dp"

OUTPUT_DIR은 학습된 모델 및 실험 결과가 저장될 위치를 나타내며, DATA_DIR은 학습 및 평가를 위한 데이터의 위치를 나타냄

학습 및 평가를 위해서는 다음의 명령어를 실행하면 됨

- 주요 파라미터

model_name_or_path: 학습에 사용될 Huggingface에 배포된 모델의 URL을 URL을 넣으면 됨. 현재는 monologg/koelectra-base-v3-discriminator를 사용

learning_rate: 모델의 학습률을 지정

train_batch_size: 학습을 위한 배치 사이즈를 나타내며, 사용하는 GPU의 메모리에 맞게 설정. 현재는 RTX-3090(24Gb)에 맞게 설정됨

warmup_ratio: 지정한 비율만큼 처음 학습을 시작했을 때, 훨씬 작은 Learning Rate로 학습을 하도록 하는 파라미터


- 학습 명령어

python dp_main.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 4


- 평가 명령어

python dp_main.py evaluate --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 3 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 36



- 테스트 명령어

python dp_main.py test --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 3 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 36




## 학습 결과
UAS 95.76 LAS 92.55

학습 후 model_output 폴더의 Metrics.csv에 매 validation step마다 저장이 되어있음
왼쪽부터 Micro UAS, Macro UAS, Micro LAS, Macro LAS
정확한 결과는 두번째 네번째인 Macro UAS와  Macro LAS를 확인됨

## 결과 분석
- 성능 향상

![성능결과(8-05)](https://user-images.githubusercontent.com/66815358/182917840-74f95216-ac07-4523-86f1-661276c9be6d.png)
   - 과제 시작(1월 대비) 7월 보고 기준 UAS 5.52, LAS 7.42% 향상
- 타 모델들과 성능 비교

![성능비교(8-05)](https://user-images.githubusercontent.com/66815358/182920071-79ac233f-0ce9-438d-b225-daab8ef2df81.png)
   - UAS에서 SOTA를 달성하였으며, 기존 최고성능 모델대비 UAS 0.96가 높음
- 학습 속도
![학습속도](https://user-images.githubusercontent.com/66815358/182920925-268e142a-a8c7-4488-b281-3f8f524f9d8b.png)
   - 학습데이터 1/8
   - 학습시간 : 약 11배 향상
   - 성능: UAS 0.95% 향상

## 모델 특징
- 데이터에 태깅 에러가 있음에도 잘 예측할 수 있다 : 오류가 있어도 됨

- 학습시간을 줄일 수 있다 : 

- 에러 분석 및 설명이 가능하다 : 

- 다른모델에도 적용할 수 있다(일반화) : 


## 학습된 모형 배포

http://pnuailab.synology.me/sharing/irbmcpyIY


## 데모 실행 버전(Inference 예정)

추후 업데이트 할 예정입니다(8월 중 예상)

## Reference

https://www.koreascience.or.kr/article/CFKO202130060562801.pdf

https://github.com/KLUE-benchmark/KLUE


## Announce

Inference : 규칙 적용 Layer로 전처리 과정과 모델 파라미터가 복잡해져 추가 튜닝과정에 있으며, 8월 중 업로드할 예정입니다.
