#-*-coding:utf-8-*-
from transformers import AutoTokenizer, BertForSequenceClassification, logging
logging.set_verbosity_error()
import re
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings(action='ignore')

###########################################################################################################

# label
#1 : small, 2 : big, 3 : region
big_label = ['정치', '경제', '사회', '문화', '국제', '스포츠', 'IT_과학']

small_label = ['국회_정당', '북한', '선거', '외교', '청와대', '행정_자치', '골프', '농구_배구', '야구_메이저리그',
               '야구_일본프로야구', '올림픽_아시안게임', '축구_월드컵', '축구_한국프로축구', '축구_해외축구',
               '교육_시험', '날씨', '노동_복지', '미디어', '사건_사고', '여성', '의료_건강', '장애인', '환경',
               '미술_건축', '방송_연예', '생활', '요리_여행', '음악', '전시_공연', '종교', '출판', '학술_문화재',
               '러시아', '미국_북미', '아시아', '유럽_EU', '일본', '중국', '중남미', '중동_아프리카', '국제경제',
               '금융_재테크', '무역', '반도체', '부동산', '산업_기업', '서비스_쇼핑', '외환', '유통', '자동차',
               '자원', '증권_증시', '취업_창업', '과학', '모바일', '보안', '인터넷_SNS', '콘텐츠']

region_label = ['강원', '경기', '경남', '경북', '광주', '대구', '대전', '부산', '울산', '전남', '전북',
                '제주', '지역일반', '충남', '충북']

BS_label = {'정치': ['국회_정당', '북한', '선거', '외교', '정치일반', '청와대', '행정_자치'],
             '스포츠': ['골프', '농구_배구', '스포츠일반', '야구_메이저리그', '야구_일본프로야구', '올림픽_아시안게임', '축구_월드컵', '축구_한국프로축구', '축구_해외축구'],
             '사회': ['교육_시험', '날씨', '노동_복지', '미디어', '사건_사고', '사회일반', '여성', '의료_건강', '장애인', '환경'],
             '문화': ['문화일반', '미술_건축', '방송_연예', '생활', '요리_여행', '음악', '전시_공연', '종교', '출판', '학술_문화재'],
             '국제': ['국제일반', '러시아', '미국_북미', '아시아', '유럽_EU', '일본', '중국', '중남미', '중동_아프리카'],
             '경제': ['경제일반', '국제경제', '금융_재테크', '무역', '반도체', '부동산', '산업_기업', '서비스_쇼핑', '외환', '유통', '자동차', '자원', '증권_증시', '취업_창업'],
             'IT_과학': ['IT_과학일반', '과학', '모바일', '보안', '인터넷_SNS', '콘텐츠'],
             '지역': ['강원', '경기', '경남', '경북', '광주', '대구', '대전', '부산', '울산', '전남', '전북', '제주', '지역일반', '충남', '충북']}

big_label2id = {label: i for i, label in enumerate(big_label)}
big_id2label = {i: label for label, i in big_label2id.items()}

small_label2id = {label: i for i, label in enumerate(small_label)}
small_id2label = {i: label for label, i in small_label2id.items()}

region_label2id = {label: i for i, label in enumerate(region_label)}
region_id2label = {i: label for label, i in region_label2id.items()}

###########################################################################################################

# tokenizer 및 model 불러오기
tokenizer = AutoTokenizer.from_pretrained("kpfbert")
kpf_model1 = BertForSequenceClassification.from_pretrained("kpf-BERT-cls-L1")
kpf_model2 = BertForSequenceClassification.from_pretrained("kpf-BERT-cls-L2")
kpf_model3 = BertForSequenceClassification.from_pretrained("kpf-BERT-cls-L3")

kpf_model1.to("cuda")
kpf_model2.to("cuda")
kpf_model3.to("cuda")

###########################################################################################################

def cls_predict(text):
    if not text: return
    text = clean_text(text)
    test_tokenized = tokenizer(text, padding='max_length', truncation=True, stride=300, max_length=512, return_tensors="pt",
              return_token_type_ids=False, return_attention_mask=False, return_offsets_mapping=False)['input_ids']
    input = test_tokenized
    input = test_tokenized.to("cuda")

    output1 = kpf_model1(input)
    output2 = kpf_model2(input)
    output3 = kpf_model3(input)

    _pred1, indices1 = torch.sort(F.softmax(output1[0][0]), descending=True)
    _pred2, indices2 = torch.sort(F.softmax(output2[0][0]), descending=True)
    pred3 = torch.argmax(F.softmax(output3[0][0]))
    _pred3 = torch.max(F.softmax(output3[0][0]))

    small_val = _pred1[:3]
    small_idx = indices1[:3]

    big_val = _pred2[:3]
    big_idx = indices2[:3]

    region_val = _pred3
    region_idx = pred3

    result = dict()

    check = 0
    for s in small_idx:
        if small_label[s] in BS_label[big_label[big_idx[0]]]:
            _small = s
            check = 1
            break

    if not check:
        result['big_cls'] = big_label[big_idx[0]]
        result['small_cls'] = big_label[big_idx[0]] + "일반"
        result['region_cls'] = region_label[region_idx]
    else:
        result['big_cls'] = big_label[big_idx[0]]
        result['small_cls'] = small_label[_small]
        result['region_cls'] = region_label[region_idx]

    return result

def clean_text(text):
    text = re.sub("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z-.]+)","", text)
    text = re.sub("[\r\n\xa0]", "", text)
    text = re.sub("(\.\s+[ㄱ-ㅎ가-힣]+\s[기]+[자]+)", "", text)
    text = re.sub("[^\w\s^.]", " ", text)
    return text


if __name__ == "__main__":
    text = """
울산지역 제조업 체감경기가 부진한 수준에 머물렀다.

한국은행 울산본부가 30일 발표한 1월 울산지역 기업 체감경기를 보면 제조업 업황 기업경기실사지수(BSI)는 58로 전월(66)보다 8포인트 하락했다.

석유화학과 자동차 업종에서 체감경기가 악화한 영향으로 풀이됐다.

1월 비제조업 업황 BSI는 63로 전월(64)보다 1포인트 떨어졌다.

도소매업, 건설장비 임대업 등 사회서비스업을 중심으로 체감경기가 나빠진 것으로 나타났다.
    """
    print(cls_predict(text))