# BIGKINDS-LAB


한국언론진흥재단과 (주)포스웨이브에서 제작한 KPF-BERT fine-tunning 모델을 제공합니다.
해당 모델과 알고리즘은 [빅카인즈랩](lab.bigkinds.or.kr)에서 제공되는 python 분석환경에서 API로 호출하여 사용할 수 있습니다.

- 뉴스분류 모델과 개체명인식 모델은 [KPF huggingface](https://huggingface.co/KPF)에 공개되어 있습니다.
- 뉴스분류 모델은 `KPF-BERT-CLS/cls_module.py`를 통해 사용할 수 있습니다.
- 개체명인식 모델은 `KPF-BERT-NER/ner_module.py`를 통해 사용할 수 있습니다. 

## 제공 API

- 자동분류 [KPF-BERT-CLS](https://github.com/KPF-bigkinds/BIGKINDS-LAB/tree/main/KPF-BERT-CLS)
- 개체명인식 [KPF-BERT-NER](https://github.com/KPF-bigkinds/BIGKINDS-LAB/tree/main/KPF-BERT-NER)
- 기사요약 [KPF-BERTSum](https://github.com/KPF-bigkinds/BIGKINDS-LAB/tree/main/KPF-BERTSum)
- 키워드추출 [KPF-KeyBERT](https://github.com/KPF-bigkinds/BIGKINDS-LAB/tree/main/KPF-KeyBERT)
- 형태소분석기 [Bareun-NLP_PY](https://github.com/KPF-bigkinds/Bareun-NLP_PY)


## 분석환경 KPF-BERT API 사용법

### API 기본 사용법

```
# library import
import requests
import json

url = http://api.bigkindslab.or.kr:5002/get_summary # API를 불러올 url
headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*’} # header 정보

body = dict() # request body 부분
body[‘text’] = text

response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent=＂\t＂).encode(＇utf-8’))
res_json = response.json()
print(res_json) # response 결과 확인
```

### 뉴스 데이터셋 활용

- 빅카인즈랩 제공되는 뉴스 데이터셋의 활용 방법입니다.

```
# library import
import pandas as pd

# 사용할 데이터의 csv명 입력 필요
bigkinds = pd.read_csv("230307090858.csv") # 뉴스데이터셋 데이터프레임으로 변환

bigkinds['dataContent'] = bigkinds['dataContent'].str.replace(pat=r'[^A-Za-z가-힣ㄱ-ㅎ0-9]', repl=r' ', regex=True) # 뉴스 본문 정제
sentences = bigkinds['dataContent']
print(sentences) # 20,000건 뉴스본문 데이터 결과 확인
print(sentences[0]) # 뉴스 한 건에 대한 뉴스본문 데이터
text = sentences[0] # 텍스트 데이터를 사용하여 request에 활용 가능
```

---

### 1. KPF-BERT-CLS
- url = " http://api2.bigkindslab.or.kr:5002/get_cls"
- request
  - text : (String) 뉴스 분류에 사용될 기사 본문 데이터
- response
  - result : (JSON)
    - big_cls : (String) 기사 대분류
    - small_cls : (String) 기사 소분류
    - region_cls : (String) 기사 지역분류

```
# library import
import requests
import json

url = "http://api2.bigkindslab.or.kr:5002/get_cls"
headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
body = dict()
body['text'] = text
response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent="\t").encode('utf-8'))
res_json = response.json()
print(res_json)
```

- 실행결과

```
{'result': {'big_cls': '문화', 'region_cls': '제주', 'small_cls': '출판'}}
```

### 2. KPF-BERT-NER
- url = “http://api2.bigkindslab.or.kr:5002/get_ner"
- request
  - text : (String) 개체명 인식에 사용될 기사 본문 데이터
- response
  - result : (Array)
    - word : (String) 단어
    - label : (String) 개체 라벨
    - desc : (String) 개체 라벨명

```
# library import
import requests
import json

url = "http://api2.bigkindslab.or.kr:5002/get_ner"
headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
body = dict()
body['text'] = text
response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent="\t").encode('utf-8'))
res_json = response.json()
print(res_json)
```

- 실행결과

```
{'result': [{'desc': '교통수단, 놀이기구', 'label': 'AF_TRANSPORT', 'word': '자동차'}
	, {'desc': '직업', 'label': 'CV_OCCUPATION', 'word': '택시기사'}, {'desc': '순서', 'label': 'QT_ORDER', 'word': '1심'}
	, {'desc': '법률', 'label': 'OGG_LAW', 'word': '부산고법'}, {'desc': '순서', 'label': 'QT_ORDER', 'word': '재판1'}
	, {'desc': '직위, 직책', 'label': 'CV_POSITION', 'word': '부장판사'}
	, ……]}
```

### 3. KPF-BERTSum API

- url = "http://api.bigkindslab.or.kr:5002/get_summary"
- request
    - sentences : (String) 기사요약에 사용될 기사 본문 데이터
- response
      - result : (Array) 기사요약 문장 3개

```
# library import
import requests
import json

url = "http://api.bigkindslab.or.kr:5002/get_summary"
headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
body = dict()
body['sentences'] = text
response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent="\t").encode('utf-8'))
res_json = response.json()
print(res_json)
```

- 실행결과

```
{'result’: ['부산고법 울산재판부 형사1부 부장판사 박해빈 는 업무상 과실치사 혐의로 기소된 택시기사 A 69 씨에게 금고 1년 집행유예 2년을 선고했다고 13일 밝혔다 재판부는 A씨에게 40시간의 사회봉사도 명령했다'
    , '자동차전용도로 갓길에서 하차한 만취한 승객이 사망한 사건과 관련 항소심 재판부가 택시기사에게 유죄를 선고했다 1심 재판부는 무죄를 선고했었다'
    , 'A씨는 2019년 4월 18일 밤 술에 취한 손님 B 당시 27세 씨를 울산시 울주군 청량읍의 한 자동차전용도로 갓길에 내려주고 갔다']}
```


### 4. KPF-KeyBERT

- url = “http://api.bigkindslab.or.kr:5002/get_keyword"
- request
  - text : (String) 키워드 추출에 사용될 기사 본문 데이터
- response
  - result : (Array) 기사로부터 추출된 키워드 5개

```
# library import
import requests
import json

url = "http://api.bigkindslab.or.kr:5002/get_keyword"
headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
body = dict()
body['text'] = text
response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent="\t").encode('utf-8'))
res_json = response.json()
print(res_json)
```

- 실행결과

```
{'result': ['택시', '갓길', '자동차', '치사', '울산재판부']}
```

### 5. KPF-형태소분석기

- url = "http://api.bigkindslab.or.kr:5002/get_tag"
- request
  - text : (String) 형태소 분석에 사용될 기사 본문 데이터
- response
  - result : (Array)
    - word : (String) 단어
    - pos : (String) 형태소 태그코드
    - desc : (String) 형태소 태그명
  
```
# library import
import requests
import json

url = "http://api.bigkindslab.or.kr:5002/get_tag"
headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
body = dict()
body['text'] = text
response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent="\t").encode('utf-8'))
res_json = response.json()
print(res_json)
```

- 실행 결과

```
{'result': [{'desc': '일반 명사', 'pos': 'NNG', 'word': '자동차'}
	, {'desc': '일반 명사', 'pos': 'NNG', 'word': '전용'}
	, {'desc': '일반 명사', 'pos': 'NNG', 'word': '도로'}
	, {'desc': '일반 명사', 'pos': 'NNG', 'word': '갓길'}
	, {'desc': '부사격 조사', 'pos': 'JKB', 'word': '에서'}
	, {'desc': '동사', 'pos': 'VV', 'word': '하차하'}
	, {'desc': '관형형 전성 어미', 'pos': 'ETM', 'word': ‘ㄴ’}
	, ......]}
```

