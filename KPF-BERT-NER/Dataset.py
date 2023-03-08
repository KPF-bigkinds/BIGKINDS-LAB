from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List
from torch.utils.data import Dataset
import torch
import label

max_length = 128
tokenizer = AutoTokenizer.from_pretrained("kpfbert")
#########################################################################################################################
"""
    Dataset 클래스.
    - 데이터셋을 생성. 학습시 해당 데이터셋의 데이터를 collate_fn을 사용하여 변환 후 불러옴.
    dataset : 문장들, BIO 표기법으로 표시된 토큰들과 그 위치를 표기.
    tokenizer : kpf-bert tokenizer
    shuffle : 데이터셋을 셔플할지 여부.
"""
############################################################################################################################


class NerDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        examples: List,
        shuffle: bool = False,
        **kwargs
    ):
        self.dataset = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]

        return instance

#################################################################################################################################
"""
    데이터를 전처리하고 분류하는 함수.
    말뭉치 데이터를 받아 학습에 필요한 input 형태로 변환.
    sentence : 문장 (ex. "아디다스의 대표 운동화 '스탠스미스'가 연간 800만 켤레 팔리는 것과 비교하면 놀랄 만한 실적이다")
    token_label : 토큰의 클래스 (ex. ['B-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'O', 'O', 'O', 'O', 'B-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS',
                                    'I-AFW_OTHER_PRODUCTS', 'O', 'O', 'O', 'B-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    char_label : 단어의 클래스 <BIO표기법> (ex. ['B-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'I-OGG_ECONOMY', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                                'B-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS', 'I-AFW_OTHER_PRODUCTS',
                                                'O', 'O', 'O', 'O', 'O', 'O', 'B-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT', 'I-QT_COUNT',
                                                'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    offset_mapping : 토큰의 위치정보 (ex. [(0, 0), (0, 2), (2, 3), (3, 4), (4, 5), (6, 8), (9, 12), (13, 14), (14, 16), (16, 17), (17, 19), (19, 20), (20, 21), (22, 24), (25, 28),
                                    (28, 29), (30, 32), (33, 35), (35, 36), (37, 38), (38, 39), (40, 42), (42, 44), (45, 47), (48, 50), (51, 53), (53, 54), (54, 55), (55, 56), (0, 0)])
"""
#################################################################################################################################
def load_data(data, tokenizer, max_length: int = 128):
    
    data_list = []
    cnt = 0
    #데이터 전처리 및 분류
    for i in range(len(data['document'])):
        for j in range(len(data['document'][i]['sentence'])):
            token_labels = []
            sentence = data['document'][i]['sentence'][j]['form']
            char_labels = ["O"] * len(sentence)
            
            #BIO 표기법
            for k in range(len(data['document'][i]['sentence'][j]['NE'])):
                begin = data['document'][i]['sentence'][j]['NE'][k]['begin']
                end = data['document'][i]['sentence'][j]['NE'][k]['end']
                
                for q in range(begin, end):
                    if q == begin:
                        char_labels[q] = "B-" + data['document'][i]['sentence'][j]['NE'][k]['label']
                    else:
                        char_labels[q] = "I-" + data['document'][i]['sentence'][j]['NE'][k]['label']
            
            offset_mappings = tokenizer(sentence, max_length=max_length, return_offsets_mapping=True, truncation=True)["offset_mapping"]
            for offset in offset_mappings:
                start, end = offset
                if start == end == 0:
                    continue
                token_labels.append(char_labels[start])
                
            #model input 형태
            instance = {
                "sentence": sentence,
                "token_label": token_labels,
                "char_label": char_labels,
                "offset_mapping": offset_mappings
            }
            data_list.append(instance)
    return data_list

####################################################################################################
"""
    model에 input으로 들어갈 수 있게 데이터를 변환
    딥러닝에 사용할 수 있도록 tensor 형태나 자료형을 변환
 """
####################################################################################################
def collate_fn(input_examples):
    input_texts, input_labels_str = [], []
    for input_example in input_examples:
        text, label_strs = input_example["sentence"], input_example["token_label"]
        input_texts.append(text)
        input_labels_str.append(label_strs)

    encoded_texts = tokenizer.batch_encode_plus(
        input_texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
        return_offsets_mapping=True
    )
    input_ids = encoded_texts["input_ids"]
    token_type_ids = encoded_texts["token_type_ids"]
    attention_mask = encoded_texts["attention_mask"]

    len_input = input_ids.size(1)
    input_labels = []
    for input_label_str in input_labels_str:
        input_label = [label.label2id[x] for x in input_label_str]
        if len(input_label) > max_length - 2:
            input_label = input_label[:max_length - 2]
            input_label = [-100] + input_label + [-100]
        else:
            input_label = (
                [-100] + input_label + (max_length - len(input_label_str) - 1) * [-100]
            )
        input_label = torch.tensor(input_label).long()
        input_labels.append(input_label)

    input_labels = torch.stack(input_labels)
    return input_ids, token_type_ids, attention_mask, input_labels