# KPF-BERT-CLS

## 모델 소개

### KPF-BERT-CLS


한국언론진흥재단이 개발한 kpf-BERT 모델을 기반으로 CLS(Classification) task를 수행할 수 있는 kpf-BERT-cls 모델을 설계 및 개발한다.

- 본 예제에 사용된 kpf-BERT는 [kpfBERT](https://github.com/KPFBERT/kpfbert)에 공개되어 있다.

- 본 예제에서는 대분류, 지역을 제외한 대분류들의 세분류, 지역 세분류로 구분하여 데이터를 학습한다.

학습데이터는 기사내용과 분류명을 넣어 제작하였다. 분류명은 아래의 분류체계를 따르며, 기사내용 + 대분류(지역제외) 데이터셋, 기사내용 + 세분류(지역제외) 데이터셋, 기사내용 + 지역세분류 데이터셋으로 나누어 학습을 진행했다.

![img](https://user-images.githubusercontent.com/87846939/221474119-7701e4e4-fe73-4b74-8f55-58d0853e5639.png)


한국언론진흥재단이 개발한 kpf-BERT를 기반으로 classification layer를 추가하여 kpf-BERT-cls 모델을 개발한다. kpf-BERT-cls 모델은 기사를 입력받아 kpf-BERT 토크나이저를 사용하여 해당 기사가 어느 클래스에 속하는지 예측한다.
기본 BERT 모델의 구조와 토크나이저는 아래의 그림과 같다.

![img_2](https://user-images.githubusercontent.com/87846939/221474169-552bba7c-0a05-4f3d-a90e-2ad8f9f69cba.png)

![img_3](https://user-images.githubusercontent.com/87846939/221474197-2b588cea-4d73-4caf-b451-b52a10ef966d.png)

BERT는 입력 길이의 제한으로 512 subword 이하의 값만 입력받을 수 있다. 기사의 특성상 인터뷰 등의 글은 512 subword보다 긴 것이 대부분이다. 이를 해결하기 위해 본 과제에서는 stride를 주어 독립적으로 문서의 조각들을 처리한다.


![img_1](https://user-images.githubusercontent.com/87846939/221474214-4e760c55-ba53-4e08-9154-65c73afabca6.png)

kpf-BERT-cls는 대분류 예측 모델, 소분류 예측 모델, 지역 소분류 예측 모델로 구성되어 있다. 대분류/소분류 예측 모델은 top-3 결과를 출력한다.


![img_4](https://user-images.githubusercontent.com/87846939/221474226-fb68c3aa-b45a-4bdf-9c10-a6c98b6451e8.png)



---
## 실행하기 위한 환경

1. 라이브러리

    ```
    python==3.7
    PyTorch==1.7.1
    tqdm==4.64.0
    transformers==4.24.0
    seqeval==0.0.12
    
    그 외 pytorch 등 머신러닝을 위한 기본적인 패키지(소스 내 import 참조)
    ```
    
---
## Usage

1. BERT

    kpfBERT 모델을 다운받아 아래의 디렉토리에 넣어주고 fine-tuning을 진행한다.

    - `kpfbert` 다운받은 kpfBERT 화일들이 위치할 곳
 

2. Data Processing

    모델 전처리 과정은 학습 데이터(csv 파일)를 불러오고 전처리 과정을 거친 후 필요한 정보로 파싱한다. 대분류 모델은 학습 데이터에서 지역 기사들을 제거한 후 대분류 코드로 구분하여 파싱한다. 세분류 모델은 대분류 모델과 마찬가지로 학습 데이터에서 지역 기사들을 제거한 후 세분류 코드로 구분하여 파싱한다. 지역 세분류 모델은 학습 데이터에서 지역 데이터를 뽑아 지역 세분류 코드로 구분하여 파싱한다.    

    `Dataset.py`에서 데이터 전처리 과정을 진행한다.
   - ClsDataset : cls dataset 클래스입니다. (torch의 dataset 라이브러리 사용, 해당 문서 참고)
   - sampling_func : 데이터셋에서 클래스별 동일한 비율로 뽑는 함수입니다. 
   - clean_text : 데이터셋 전처리 함수 (기자, 특수 문자, 이메일 등 제거)
   

3. Train
    ```
   python train.py -v (b or s or r) -s TRAIN_FILE -o MODEL_NAME 
   (ex. python train.py -v b -s dataset/NXEL2102203310.csv -o kpf-BERT-cls)
   ```
   - -v b : 대분류
   - -v s : 소분류
   - -v r : 지역분류
   - 학습할 데이터셋과 학습 결과 모델의 이름을 입력.
   - `kpf-BERT-cls` : KPF-BERT-CLS 모델의 저장 위치
   - `train.py` : 학습 관련 코드. 
   - 실행에 필요한 파일 : label.py, config.py, Dataset.py, kpfbert 폴더가 있어야 함.
   - input : dataset, model name
   - output : trained model

   ![img_5](https://user-images.githubusercontent.com/87846939/221474279-21a10884-e059-4f72-924c-2583930e2d5e.png)
      
   학습데이터의 입력값과 출력값
   

4. Predictions

    `cls_module.py`의 `cls_predict(text)`를 사용하여 `text`에 기사 전문을 넣어 출력값을 `big_cls`, `small_cls`, `region_cls` json 형태로 추출할 수 있다.

    예측 모듈을 사용하기 위해 필요한 모델
    - `kpf-BERT-cls-L1` : 대분류 예측 모델
    - `kpf-BERT-cls-L2` : 세분류 예측 모델
    - `kpf-BERT-cls-L3` : 지역 세분류 예측 모델

    예시)
    
    입력문장

    ```
    울산지역 제조업 체감경기가 부진한 수준에 머물렀다.
    한국은행 울산본부가 30일 발표한 1월 울산지역 기업 체감경기를 보면 제조업 업황 기업경기실사지수(BSI)는 58로 전월(66)보다 8포인트 하락했다.
    석유화학과 자동차 업종에서 체감경기가 악화한 영향으로 풀이됐다.
    1월 비제조업 업황 BSI는 63로 전월(64)보다 1포인트 떨어졌다.
    도소매업, 건설장비 임대업 등 사회서비스업을 중심으로 체감경기가 나빠진 것으로 나타났다.
    BSI가 100을 넘으면 경기가 호전된 것으로 보는 기업이 더 많다는 뜻이고, 100 미만이면 그 반대를 의미한다.
    1월 제조업체 경영 애로사항은 불확실한 경제 상황(22.5%), 인력난과 인건비 상승(20.1%), 자금 부족(14.3%) 등으로 조사됐다.
    비제조업체는 인력난과 인건비 상승(20.5%), 불확실한 경제 상황(15.0%), 내수 부진(13.4%) 순이었다.
    한국은행 울산본부는 이달 10∼17일 제조업 131개, 비제조업 134개 등 265개 업체를 대상으로 조사했다.
    1월 전국 제조업과 비제조업 업황 BSI는 각각 66과 71을 기록, 모두 울산보다 높았다.

   ```
    결과값    

    ```
   {'big_cls': '경제', 'small_cls': '산업_기업', 'region_cls': '울산'}
   ```
