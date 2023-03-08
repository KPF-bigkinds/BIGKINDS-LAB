import config
import torch
import numpy as np
from transformers import BertConfig, AutoTokenizer, BertForTokenClassification, AdamW, logging
logging.set_verbosity_error()
from seqeval.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
import json, sys
from tqdm import tqdm, trange
import Dataset
import label

############################################################################################################
"""
  train_epoch(epoch, model, dataloader, optimizer, con) : 학습 함수
  - 반복문을 돌면서 데이터를 학습.
  - 데이터셋에서 데이터들을 불러와 input 형태에 맞게 가공후 원본 데이터와 비교하며 학습.
"""
###########################################################################################################

def train_epoch(epoch, model, dataloader, optimizer, con):

    model.train()
    total_loss = 0.0
    
    #학습 진행을 보여주는 library
    tepoch = tqdm(dataloader, unit="batch", position=1, leave=True)
    for batch in tepoch:
        tepoch.set_description(f"Train")
        model.zero_grad()
        
        #input data
        input_ids = batch[0].to(con.device)
        token_type_ids = batch[1].to(con.device)
        attention_mask = batch[2].to(con.device)
        labels = batch[3].to(con.device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        
        #output
        outputs = model(**inputs)

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

#valid
################################################################################################
"""
    valid_epoch(epoch, dataloader, model, con) : 검증 함수.
    - 반복문을 돌면서 학습한 모델을 검증.
    - 과정은 학습과 동일함.
"""
################################################################################################

def valid_epoch(epoch, dataloader, model, con):
    total_loss = 0.0

    model.eval()
    all_token_predictions = []
    all_token_labels = []

    tepoch = tqdm(dataloader, unit="batch", leave=False)
    for batch in tepoch:
        tepoch.set_description(f"Valid")
        with torch.no_grad():
            input_ids = batch[0].to(con.device)
            token_type_ids = batch[1].to(con.device)
            attention_mask = batch[2].to(con.device)
            labels = batch[3].to(con.device)
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            token_predictions = logits.argmax(dim=2)
            token_predictions = token_predictions.detach().cpu().numpy()

            for token_prediction, label in zip(token_predictions, labels):
                filtered = []
                filtered_label = []
                for i in range(len(token_prediction)):
                    if label[i].tolist() == -100:
                        continue
                    filtered.append(label.id2label[token_prediction[i]])
                    filtered_label.append(label.id2label[label[i].tolist()])
                assert len(filtered) == len(filtered_label)
                all_token_predictions.append(filtered)
                all_token_labels.append(filtered_label)

        tepoch.set_postfix(loss=loss.mean().item())

    token_f1 = f1_score(all_token_labels, all_token_predictions, average="macro")
    return total_loss / len(dataloader),  token_f1

############################################################################################
"""
    test_epoch(dataloader, model, con) : 테스트 함수.
    학습한 모델이 얼마의 성능을 가지는지 테스트.
    과정은 학습과 같음.
"""
############################################################################################
def test_epoch(dataloader, model, con):
    total_loss = 0.0

    model.eval()
    all_token_predictions = []
    all_token_labels = []

    tepoch = tqdm(dataloader, unit="batch")
    for batch in tepoch:
        tepoch.set_description(f"Test")
        with torch.no_grad():
            
            input_ids = batch[0].to(con.device)
            token_type_ids = batch[1].to(con.device)
            attention_mask = batch[2].to(con.device)
            labels = batch[3].to(con.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            token_predictions = logits.argmax(dim=2) # logits
            token_predictions = token_predictions.detach().cpu().numpy()

            for token_prediction, label in zip(token_predictions, labels):
                filtered = []
                filtered_label = []
                for i in range(len(token_prediction)):
                    if label[i].tolist() == -100:
                        continue
                    filtered.append(label.id2label[token_prediction[i]])
                    filtered_label.append(label.id2label[label[i].tolist()])
                assert len(filtered) == len(filtered_label)
                all_token_predictions.append(filtered)
                all_token_labels.append(filtered_label)

            tepoch.set_postfix(loss=loss.mean().item())
    
    #학습 결과에 대한 보고서 작성
    token_result = classification_report(all_token_labels, all_token_predictions)
    token_f1 = f1_score(all_token_labels, all_token_predictions, average="macro")

    print(token_result)

    tepoch.set_postfix(loss=total_loss / len(dataloader), token_f1=token_f1)
    return total_loss / len(dataloader), token_f1

###################################################################################################
"""
    set_optimizer(model, con) : 학습에 사용하는 옵티마이저
    AdamW를 사용.
"""
###################################################################################################
def set_optimizer(model, con):
    optimizer_grouped_parameters = [
        {'params': model.bert.parameters(), 'lr': con.learning_rate / 100 },
        {'params': model.classifier.parameters(), 'lr': con.learning_rate }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=con.learning_rate, eps=con.adam_epsilon)
    
    return optimizer

###################################################################################################
"""
    load_dataset(data, tokenizer, con) : 학습, 검증, 테스트 데이터셋을 생성.
"""
###################################################################################################
def load_dataset(data, tokenizer, con):
    #make data to dataset
    dataset = Dataset.load_data(data, tokenizer)
    
    #train dataset
    index = int(len(dataset) * 0.7)
    train_dataset = Dataset.NerDataset(
        tokenizer,
        dataset[:index],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Dataset.collate_fn
    )
    
    #valid dataset
    index2 = int(len(dataset) * 0.9)
    valid_dataset = Dataset.NerDataset(
        tokenizer,
        dataset[index:index2],
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Dataset.collate_fn
    )
    
    #test dataset
    test_dataset = Dataset.NerDataset(
        tokenizer,
        dataset[index2:],
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Dataset.collate_fn
    )
    
    return train_dataloader, valid_dataloader, test_dataloader

###################################################################################################
"""
    train(model, train_dataloader, valid_dataloader, test_dataloader, con) : 모델 학습 함수.
    사전에 정한 epoch 만큼 반복하면서 학습, 검증을 반복.
    모델을 학습한 이후 테스트를 수행.
"""
###################################################################################################
def train(model, train_dataloader, valid_dataloader, test_dataloader, con):
    model.to("cuda")

    best_f1 = 0.0
    best_model = None 

    tepoch = trange(con.epoch, position=0, leave=True)
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        #train
        train_loss = train_epoch(epoch, model, train_dataloader, optimizer, con)
        #valid
        valid_loss, token_f1 = valid_epoch(epoch, valid_dataloader, model, con)

        if best_f1 < token_f1:
            best_f1 = token_f1
            best_model = model

        tepoch.set_postfix(valid_f1=token_f1)
    #test
    test_loss, token_f1 = test_epoch(test_dataloader, model, con)
    
    return best_model

###################################################################################################
"""
    save_model(model, model_name) : 추론 함수를 실행하는 메인 함수.
    - 입력 형태 : train -s TRAIN_FILE -o MODEL_NAME
    - 학습할 데이터셋과 학습 결과 모델의 이름을 입력.
"""
###################################################################################################
def save_model(model, model_name):
    model.save_pretrained(model_name)

if __name__ == "__main__":
   
    if len(sys.argv) < 5:
        print("학습할 데이터셋과 모델명을 입력해주세요.")
        sys.exit()
    if sys.argv[1] != "-s":
        print("train -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[3] != "-o":
        print("train -s TRAIN_FILE -o MODEL_NAME")
        sys.exit()
    if sys.argv[4] == "kpfbert":
        print("kpfbert 외의 다른 이름을 입력해주세요")
        sys.exit()
        
    print("Start train!")
    
    #학습할 모델의 환경을 설정한다.
    _config = config.Config()

    max_length = _config.max_seq_len
    batch_size = _config.batch_size

    #랜덤을 seed 에 고정
    torch.manual_seed(_config.seed)
    np.random.seed(_config.seed)

    bertconfig = BertConfig.from_pretrained(_config.model_name, num_labels=len(label.label2id))
    bertconfig.update(_config.__dict__)
    
    model_path = "kpfbert" # Bert 바이너리가 포함된 디렉토리

    #load model & tokenizer
    model = BertForTokenClassification.from_pretrained(_config.model_name, config=bertconfig)
    tokenizer = AutoTokenizer.from_pretrained(_config.model_name)
    
    #load dataset
    data = json.load(open(sys.argv[2], 'rt', encoding='UTF8'))
    
    #make dataloader
    train_dataloader, valid_dataloader, test_dataloader = load_dataset(data, tokenizer, _config)
    
    #set optimizer
    optimizer = set_optimizer(model, _config)
    
    #train
    out_model = train(model, train_dataloader, valid_dataloader, test_dataloader, _config)
    
    #save model
    save_model(out_model, sys.argv[4])
    
    print("Train Done")