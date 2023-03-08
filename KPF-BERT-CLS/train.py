import config, Dataset, label
import torch
import numpy as np
from transformers import BertConfig, AutoTokenizer, BertForSequenceClassification, AdamW, logging
logging.set_verbosity_error()
from seqeval.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import pandas as pd
from tqdm import tqdm, trange

###########################################################################################################
"""
    train_epoch : 학습 함수.
    - 반복문을 돌면서 데이터를 학습.
    - 데이터셋에서 데이터들을 불러와 input 형태에 맞게 가공후 원본 데이터와 비교하며 학습.
"""
############################################################################################################

#train
def train_epoch(epoch, model, dataloader, optimizer, tokenizer, con):
    
    model.train()
    total_loss = 0.0
    
    #학습 진행을 보여주는 library
    tepoch = tqdm(dataloader, unit="batch", position=1, leave=True)
    for batch in tepoch:
        tepoch.set_description(f"Train")
        model.zero_grad()
        
        #input data
        sample = tokenizer(batch[0], padding='max_length', truncation=True, stride = con.stride,
                           max_length=con.max_seq_len, return_tensors="pt", return_token_type_ids=False,
                           return_attention_mask=False, return_offsets_mapping=False)['input_ids']
        _label = batch[1].to(con.device)
        samples = sample.to(con.device)
        labels = torch.tensor(_label)
        
        #output
        outputs = model(samples, labels=labels)

        #output loss (class)
        loss = outputs[0]
        loss.backward()
    
        #clipping & optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), con.max_grad_norm)
        optimizer.step()
        total_loss += loss.item()

        tepoch.set_postfix(loss=loss.mean().item())
    tepoch.set_postfix(loss=total_loss / len(dataloader))
    return total_loss / len(dataloader)

##############################################################################################################
"""
    valid_epoch : 검증 함수.
    - 반복문을 돌면서 학습한 모델을 검증.
    - 과정은 train_epoch 함수와 동일함.
"""
##############################################################################################################

#valid
def valid_epoch(epoch, dataloader, model, tokenizer, con, id2label):

    total_loss = 0.0
    total_len = 0
    total_correct = 0

    model.eval()
    all_token_predictions = []
    all_token_labels = []

    tepoch = tqdm(dataloader, unit="batch", leave=False)
    for batch in tepoch:
        tepoch.set_description(f"Valid")
        with torch.no_grad():
            sample = tokenizer(batch[0], padding='max_length', truncation=True, stride = con.stride,
                               max_length=con.max_seq_len, return_tensors="pt", return_token_type_ids=False,
                               return_attention_mask=False, return_offsets_mapping=False)['input_ids']
            _label = batch[1].to(con.device)
            samples = sample.to(con.device)
            labels = torch.tensor(_label)

            outputs = model(samples, labels=labels)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            labels = labels.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            all_token_labels.extend(labels)
            all_token_predictions.extend(pred)

        tepoch.set_postfix(loss=loss.mean().item())

    all_token_labels = [id2label[int(x)] for x in all_token_labels]
    all_token_predictions = [id2label[int(x)] for x in all_token_predictions]
    token_f1 = f1_score(all_token_labels, all_token_predictions, average="micro")
    print('[Epoch {}] -> F1_score : {:.4f} Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, token_f1, total_loss/len(dataloader), total_correct/total_len))
    return total_loss / len(dataloader),  token_f1

##############################################################################################################
"""
    test_epoch : 테스트 함수
    - 학습한 모델이 얼마의 성능을 가지는지 테스트함.
    - 과정은 train_epoch 함수와 동일함.
"""
##############################################################################################################

#test
def test_epoch(dataloader, model, tokenizer, con, id2label):

    total_loss = 0.0
    total_len = 0
    total_correct = 0
    
    model.eval()
    all_token_predictions = []
    all_token_labels = []

    tepoch = tqdm(dataloader, unit="batch", leave=False)
    for batch in tepoch:
        tepoch.set_description(f"Test")
        with torch.no_grad():
            
            sample = tokenizer(batch[0], padding='max_length', truncation=True, stride = con.stride,
                               max_length=con.max_seq_len, return_tensors="pt", return_token_type_ids=False,
                               return_attention_mask=False, return_offsets_mapping=False)['input_ids']
            _label = batch[1].to(con.device)
            samples = sample.to(con.device)
            labels = torch.tensor(_label)
            
            outputs = model(samples, labels=labels)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            labels = labels.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            all_token_labels.extend(labels)
            all_token_predictions.extend(pred)

            tepoch.set_postfix(loss=loss.mean().item())
    
    all_token_labels = [id2label[int(x)] for x in all_token_labels]
    all_token_predictions = [id2label[int(x)] for x in all_token_predictions]

    #학습 결과에 대한 보고서 작성
    token_result = classification_report(all_token_labels, all_token_predictions)
    token_f1 = f1_score(all_token_labels, all_token_predictions, average="micro")

    print(token_result)

    tepoch.set_postfix(loss=total_loss / len(dataloader), token_f1=token_f1)
    return total_loss / len(dataloader), token_f1

##############################################################################################################
"""
    set_optimizer(model, con) : 학습에 사용하는 최적화 도구.
    - AdamW 를 사용함.
"""
#############################################################################################################

#set optimizer
def set_optimizer(model, con):

    optimizer_grouped_parameters = [
    {'params': model.bert.parameters(), 'lr': 3e-5},
    {'params': model.classifier.parameters(), 'lr': con.learning_rate }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=con.learning_rate, eps=con.adam_epsilon, no_deprecation_warning=True)
    
    return optimizer

##############################################################################################################
"""
    load_dataset(data, con) : 학습, 검증, 테스트 데이터셋을 생성.
"""
##############################################################################################################

#load dastaset
def load_dataset(data, con):

    #train dataset
    train_data = data.groupby('LABEL', group_keys=False).apply(Dataset.sampling_func, sample_pct=0.7, _seed = con.seed)
    train_data.sort_index()
    train_dataset = Dataset.ClsDataset(
        train_data
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=con.batch_size,
        shuffle=True,
    )
    
    #valid dataset
    last_data = data.drop(data.index[train_data.index])
    valid_data = last_data.groupby('LABEL', group_keys=False).apply(Dataset.sampling_func, sample_pct=0.67, _seed = con.seed)
    valid_dataset = Dataset.ClsDataset(
        valid_data
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=con.batch_size,
        shuffle=True,
    )
    
    #test dataset
    test_data = last_data.drop(data.index[valid_data.index])
    test_dataset = Dataset.ClsDataset(
        test_data
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=con.batch_size,
        shuffle=True,
    )
    
    return train_dataloader, valid_dataloader, test_dataloader

##############################################################################################################
"""
    train(model, train_dataloader, valid_dataloader, test_dataloader, tokenizer, con, id2label) : 모델 학습 함수.
    - 사전에 정한 epoch 만큼 반복하면서 학습, 검증을 반복함.
    - 모델을 학습한 이후 테스트를 수행.
"""
##############################################################################################################

#train model
def train(model, train_dataloader, valid_dataloader, test_dataloader, tokenizer, con, id2label):

    model.to("cuda")

    best_f1 = 0.0
    best_model = None 

    tepoch = trange(con.epoch, position=0, leave=True)
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        #train
        train_loss = train_epoch(epoch, model, train_dataloader, optimizer, tokenizer, con)
        #valid
        valid_loss, token_f1 = valid_epoch(epoch, valid_dataloader, model, tokenizer, con, id2label)

        if best_f1 < token_f1:
            best_f1 = token_f1
            best_model = model

        tepoch.set_postfix(valid_f1=token_f1)
    #test
    test_loss, token_f1 = test_epoch(test_dataloader, model, tokenizer, con, id2label)
    
    return best_model

##############################################################################################################
"""
    save_model(model, model_name) : 모델 저장 함수
    - model : 저장할 모델 객체
    - model_name : 저장할 모델 이름 ( 폴더 이름으로 네이밍됨)
"""
##############################################################################################################

#save model
def save_model(model, model_name):
    model.save_pretrained(model_name)

##############################################################################################################
"""
    preprocessing_data(data) : 데이터 전처리 함수
    - data : 전처리된 데이터 객체
"""
##############################################################################################################

#preprocessing data
def preprocessing_data(data):
    #이메일, 특수 문자 등 제거
    data['NEWS_CNTS'] = data['NEWS_CNTS'].apply(lambda x: Dataset.clean_text(x))

    #내용 없는 기사 제거
    data['number_of_words'] = data['NEWS_CNTS'].apply(lambda x : len(str(x).split()))
    no_text = data[data['number_of_words'] == 0]
    data.drop(no_text.index, inplace = True)
    
    data['LABEL'] = data['NEWS_SML_SUBJ_CD'].apply(lambda x : label.str_labels[label.labels.index(str(x))])

    remove_list = ["사회일반", "경제일반", "정치일반", "문화일반", "국제일반", "IT_과학일반", "스포츠일반"] #"지역 일반은 제외"
    for r in remove_list:
        idx = label.str_labels.index(r)
        del label.str_labels[idx]
        del label.labels[idx]

    label.label2id = {label: i for i, label in enumerate(label.labels)}
    label.id2label = {i: label for label, i in label.label2id.items()}

    for i in remove_list:
        data = data.drop(data[data['LABEL'] == i].index)

    data['LABEL'] = data['LABEL'].apply(lambda x : label.label2id[label.labels[label.str_labels.index(x)]])
    data = data.reset_index(drop=True)

    return data

##############################################################################################################
"""
    main : 추론 함수를 실행하는 메인 함수.
    - 입력 형태 : train -v b -s TRAIN_FILE -o MODEL_NAME
    -v b : 대분류
    -v s : 소분류
    -v r : 지역분류
    - 학습할 데이터셋과 학습 결과 모델의 이름을 입력.
"""
##############################################################################################################

if __name__ == "__main__":
    
    if len(sys.argv) < 7:
        print("학습할 데이터셋과 모델명을 입력해주세요.")
        sys.exit()
    if sys.argv[1] != "-v":
        print("train -v (b or s or r) -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[2] != "b" and sys.argv[2] != "s" and sys.argv[2] != "r":
        print("train -v (b or s or r) -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[3] != "-s":
        print("train -v (b or s or r) -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[5] != "-o":
        print("train -v (b or s or r) -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[6] == "kpfbert":
        print("kpfbert 외의 다른 이름을 입력해주세요")
        sys.exit()
        
    print("Start train!")
    
    #학습할 모델의 환경을 설정한다.
    _config = config.Config()

    _random_state = _config.seed

    #랜덤을 seed 에 고정
    torch.manual_seed(_config.seed)
    np.random.seed(_config.seed)
    
    #load dataset
    print("Data loading..")
    data = pd.read_csv(sys.argv[4], sep = "\t")
    print("Data preprocessing..")
    data = preprocessing_data(data)

    labels = ['정치', '경제', '사회', '문화', '국제', '스포츠', 'IT_과학', '지역']

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    if sys.argv[2] == "b":
        print("Train for Large category model")
        data['LABEL'] = data['NEWS_BIG_SUBJ_NM'].apply(lambda x : label2id[x])

        ji = data[data['LABEL'] == 7]
        data.drop(ji.index, inplace = True)

        data = data[['NEWS_CNTS', 'LABEL']]
        data = data.sample(frac=1, random_state=_random_state)
        data = data.reset_index(drop=True)

        labels.pop()
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label2id.items()}

    elif sys.argv[2] == "s":
        print("Train for Samll category model")
        data['LABEL'] = data['NEWS_BIG_SUBJ_NM'].apply(lambda x : label2id[x])

        ji = data[data['LABEL'] == 7]
        data.drop(ji.index, inplace = True)

        labels = list(data['NEWS_SML_SUBJ_NM'].unique())

        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label2id.items()}

        data['LABEL'] = data['NEWS_SML_SUBJ_NM'].apply(lambda x : label2id[x])

        data = data[['NEWS_CNTS', 'LABEL']]
        data = data.sample(frac=1, random_state=_random_state)
        data = data.reset_index(drop=True)

    elif sys.argv[2] == "r":
        print("Train for Region category model")
        data['LABEL'] = data['NEWS_BIG_SUBJ_NM'].apply(lambda x : label2id[x])

        data = data[data['LABEL'] == 7]

        labels = list(data['NEWS_SML_SUBJ_NM'].unique())

        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label2id.items()}

        data['LABEL'] = data['NEWS_SML_SUBJ_NM'].apply(lambda x : label2id[x])

        data = data[['NEWS_CNTS', 'LABEL']]
        data = data.sample(frac=1, random_state=_random_state)
        data = data.reset_index(drop=True)

    bertconfig = BertConfig.from_pretrained(_config.model_name, num_labels=len(label2id))
    bertconfig.update(_config.__dict__)
    
    model_path = "kpfbert" # Bert 바이너리가 포함된 디렉토리

    #load model & tokenizer
    model = BertForSequenceClassification.from_pretrained(_config.model_name, config=bertconfig)
    tokenizer = AutoTokenizer.from_pretrained(_config.model_name)

    #make dataloader
    train_dataloader, valid_dataloader, test_dataloader = load_dataset(data, _config)
    
    #set optimizer
    optimizer = set_optimizer(model, _config)
    
    #train
    out_model = train(model, train_dataloader, valid_dataloader, test_dataloader, tokenizer, _config, id2label)
    
    #save model
    save_model(out_model, sys.argv[6])
    
    print("Train Done")