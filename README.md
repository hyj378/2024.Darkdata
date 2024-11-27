# 다크데이터 4차년도 실증
본 저장소는 "데이터 규모 확장과 정확성 향상을 위한 다크데이터 기술 개발" 과제의 4차년도 실증 코드 공유를 위해 생성되었습니다.

본 저장소의 코드는 [VFA](https://github.com/csuhan/VFA)를 기반으로 합니다.

본 저장소의 목적은 새로운 클래스를 포함한 영상 데이터셋에서 학습 성능을 높일 수 있는 소량의 데이터셋을 선정하는 방법을 공유하기 위함입니다.


## 🔨 Setup
본 저장소의 환경설정은 [VFA](https://github.com/csuhan/VFA)와 동일합니다.

추가로 [CLIP](https://github.com/openai/CLIP) 활용을 위해 아래의 설치가 필요합니다:

```bash
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```


## 🎞 Data Preparation
1. 학습에 사용할 비디오 파일을 videos 폴더에 위치시킵니다.
```bash
$ git clone {}
$ cd {}
$ mkdir videos
#├── 2024.Darkdata <- this repo
#   └── videos
#      └── parking_lot.avi
#      ├── static_pressure_machine.avi
#      └── the_entrance.avi
```
2. 데이터 가공을 위한 코드는 dataprocess에 위치합니다. 아래를 참고하여 데이터를 가공합니다:
```bash
$ cd dataprocess
# 비디오를 프레임(이미지)로 가공
$ python video2frame.py
$ cd ../
```
3. CLIP을 통해 고가치 프레임을 선정합니다:
```bash
bash select_few_shot_single_gpu.sh
```


## 🚀 Running
Data Preparation을 통해 소량의 학습할 데이터셋을 선정하였다면 아래를 실행하여 학습을 진행할 수 있습니다:

```bash
bash train_single_gpu.sh [결과를 저장할 폴더]
# ex) bash train_single_gpu.sh results-crane
```



## 🌟 Citation

```BibTeX
@InProceedings{han2023vfa,
    title     = {Few-Shot Object Detection via Variational Feature Aggregation},
    author    = {Han, Jiaming and Ren, Yuqiang and Ding, Jian and Yan, Ke and Xia, Gui-Song},
    booktitle = {Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI-23)},
    year      = {2023}
}
```
