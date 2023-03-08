#################################################################################################################
"""
Dataset 관련 코드.
ClsDataset : cls dataset 클래스입니다. (torch의 dataset 라이브러리 사용, 해당 문서 참고)
sampling_func : 데이터셋에서 클래스별 동일한 비율로 뽑는 함수입니다.
clean_text : 데이터셋 전처리 함수 (기자, 특수 문자, 이메일 등 제거)
"""
#################################################################################################################
from torch.utils.data import Dataset
import numpy as np
import re

class ClsDataset(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset.iloc[index, 0]
        label = self.dataset.iloc[index, 1]        
        return text, label

def sampling_func(data, sample_pct, _seed):
    np.random.seed(_seed)
    N = len(data)
    sample_n = int(len(data)*sample_pct) # integer
    sample = data.take(np.random.permutation(N)[:sample_n])
    return sample

def clean_text(text):
    text = re.sub("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z-.]+)","", text)
    text = re.sub("[\r\n\xa0]", "", text)
    text = re.sub("(\.\s+[ㄱ-ㅎ가-힣]+\s[기]+[자]+)", "", text)
    text = re.sub("[^\w\s^.]", " ", text)
    return text