## KT-Parser

본 저장소는 PNU AILAB KT-Parser 과제수행을 위한 Dependency Parsing model 코드 공유를 위한 것입니다.

## Dependencies

requirements.txt에 해당하는 패키지들을 설치하여 개발 환경을 구축해주셔야 합니다.

Make sure you have installed the packages listed in requirements.txt.

다음과 같은 명령어를 통해서 필요한 패키지를 설치할 수 있습니다.

```
pip install -r requirements.txt
```

모든 실험들은 파이썬 3.7 버전에서 진행되었으며, 패키지들의 Dependency를 위해서 3.7 버전(혹은 이상)의 파이썬 사용을 권장드립니다.

All expereiments are tested under Python 3.7 environment.


## 실행

파서 코드 실행을 위해 필요한 변수는 다음과 같습니다.

OUTPUT_DIR="klue_output"
DATA_DIR="data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"
task="klue-dp"

OUTPUT_DIR은 학습된 모델 및 실험 결과가 저장될 위치를 나타내며, DATA_DIR은 학습 및 평가를 위한 데이터의 위치를 나타냅니다.

학습 및 평가를 위해서는 다음의 명령어를 실행해주시면 됩니다.

주요 파라미터
model_name_or_path: 학습에 사용될 Huggingface에 배포된 모델의 URL을 URL을 넣어주시면 됩니다. 현재는 klue/roberta-base를 사용하고 있습니다.
learning_rate: 모델의 학습률을 지정합니다.
train_batch_size: 학습을 위한 배치 사이즈를 나타내며, 사용하시는 GPU의 메모리에 맞게 설정하시면 됩니다. 현재는 RTX-3090(24Gb)에 맞게 설정하였습니다.
warmup_ratio: 지정한 비율만큼 처음 학습을 시작했을 때, 훨씬 작은 Learning Rate로 학습을 하도록 하는 파라미터입니다.

학습 명령어
python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-base --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 16 --eval_batch_size 8 --max_seq_length 510 --gradient_accumulation_steps 2 --warmup_ratio 0.2 --weight_decay 0.01 --max_grad_norm 1.0 --patience 100000 --metric_key slot_micro_f1 --gpus 0 --num_workers 4

python run_klue.py test --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-base --learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 256 --metric_key las_macro_f1 --gpus 0 --num_workers 4


python run_klue.py test --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-base #--learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 256 --metric_key las_macro_f1 --gpus 0 --num_workers 4


## 학습된 모형 배포 (예정)
추후 업데이트 할 예정입니다
## 데모 실행 버전(예정)
추후 업데이트 할 예정입니다
