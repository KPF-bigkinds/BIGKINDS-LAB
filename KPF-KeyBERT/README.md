# KPF-KeyBERT

## 모델 소개

### KeyBERT

키워드 추출(keyword extraction)은 원본 문서를 가장 잘 나타내는 중요한 용어 또는 구문을 찾아내는 작업이다.
KeyBERT는 크게 4단계를 거쳐 문서에서 key word와 phrase set를 뽑아낸다.

   1. Document-level representation (by document embeddings extracted with BERT)
   2. Phrase-level representation (by word embeddings extracted for N-gram words/phrases and BERT)
   3. Use of cosine similarity to find the words/phrases that are most similar to the document
      - (optional) MMR(Maximal Marginal Relevance) or MSS(Max Sum Similarity)
   4. Extraction of words/phrases that best describe the entire document

KPF-KeyBERT는 키워드 추출을 위한 과정에서 [바른 형태소분석기](https://bareun.ai/)를 사용하여 명사를 추출하며,
문서의 유사도를 추출하기 위해 [KPF-SBERT](https://github.com/KPFBERT/kpfSBERT)를 사용합니다. 
EmbedRank에서 소개된 MMR 기법을 사용하여 키워드와 문서간의 유사도를 비교하여 키워드를 추출한다.
MMR이란 검색 엔진 내에서 본문 검색(text retreival) 관련하여 검색(query)에 따른 결과(document)의 다양성과 연관성을 control하는 방법론이다.

![img](https://user-images.githubusercontent.com/87846939/221451753-58285dc6-2fbc-47bd-9e7a-b90d3879929a.png)


### KeyBERT 관련 논문 및 사용 소스 

- 논문:  [Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](https://arxiv.org/abs/1801.04470)
- SBERT: [kpfSBERT](https://github.com/KPFBERT/kpfSBERT)
- 형태소분석기: [Bareun-NLP_PY](https://github.com/KPF-bigkinds/Bareun-NLP_PY)

---
## 실행하기 위한 환경

1. 라이브러리

    ```
    python = 3.9
    sentence_transformers==2.2.2
    bareunpy==1.6.0
    
    그 외 pytorch 등 머신러닝을 위한 기본적인 패키지(소스 내 import 참조)
    ```
    
---
## Usage

1. SBERT

    kpfSBERT 모델을 다운받아 아래의 디렉토리에 넣어 문서 .

    - `kpfSBERT` 다운받은 kpfSBERT 화일들이 위치할 곳


2. Tagger
  
   바른 형태소분석기를 사용하기 위한 설정.
   
   KPF에서 제공하는 바른 형태소분석기의 API키와 사용 접속정보는 아래와 같으며, 
   ```
   API_KEY = "koba-M3CMSHQ-ZRBUUNI-WHTY2KI-NUYJDCY" # KPF에서 제공하는 API_KEY
   tagger = Tagger(API_KEY,'175.106.88.249',5757) # KPF에서 제공하는 바른 형태소분석기
   ```
      

3. Predictions

   `keyword_module.py`의 `keyword_ext(text)`를 사용하여 `text`에 기사 전문을 넣어 KeyBERT를 통한 키워드 추출결과를 얻을 수있다.
