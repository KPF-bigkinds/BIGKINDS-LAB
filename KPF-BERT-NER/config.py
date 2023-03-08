#####################################################################################
"""
    Config 클래스.
    학습에 필요한 하이퍼파라미터.
    model_name (str): load할 kpf-bert model과 tokenizer 이름 (model과 tokenizer가 들어있는 폴더명)
    train_data (str): 학습할 데이터의 설명
    test_data (str): 테스트할 데이터의 설명
    epoch (int): 학습 횟수
    max_seq_len (int): kpf-bert-ner 에서 인식할 최대 토큰 수 (문장 길이와 연관, 최대 512)
    batch_size (int): 학습시 한번에 처리하는 데이터 수
    learning_rate (float): 학습 관련 파라미터, 학습률
    adam_epsilon (float): 학습 관련 파라미터, 최적화 변수
    max_grad_norm (float): 학습 관련 파라미터, 학습 안정화 clipping
    seed (int): 랜덤값 고정 변수, 학습 데이터셋 관련하여 고정된 랜덤값 출력
    intermediate_hidden_size (int): 모델의 hidden layer
"""
##############################################################################################################

from dataclasses import dataclass

class Config():
    model_name: str = "kpfbert"
    train_data: str = "21_150tags_EntityLinking"
    test_data: str = "21_150tags_EntityLinking"
    epoch: int = 5#15
    max_seq_len: int = 120
    batch_size: int = 4
    learning_rate: float = 5e-3
    adam_epsilon: float = 1e-8
    device: str = "cuda"
    max_grad_norm: float = 1.0
    seed: int = 1000
    intermediate_hidden_size: int = 768