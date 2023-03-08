import math
import kss
from transformers import BertTokenizer, BertModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch
from torch.nn.init import xavier_uniform_

MAX_TOKEN_COUNT = 512
N_EPOCHS = 10

BERT_MODEL_NAME = 'kpfbert-base' # kpf-BERT 경로
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

class ExtTransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2):
        super(ExtTransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, hidden_size)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, heads, d_ff, dropout)
            for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.wo = nn.Linear(hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask) 

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18) # how can i fix it to use fp16...

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


    def forward(self, x):
        inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="avg_val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="kpfBERT_Summary")

early_stopping_callback = EarlyStopping(monitor='avg_val_loss', patience=3)

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=0,
#     precision=16, #소스 수정 또는 패키지 재설치 필요... 런타임 에러.
    progress_bar_refresh_rate=30
)

class Summarizer(pl.LightningModule):

    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.max_pos = 512
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, add_pooling_layer=False)
        self.ext_layer = ExtTransformerEncoder()
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.loss = nn.BCELoss(reduction='none')
    
        for p in self.ext_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, segs, clss, labels=None): #, input_ids, attention_mask, labels=None):
        
        mask_src = ~(src == 0) #1 - (src == 0)
        mask_cls = ~(clss == -1) #1 - (clss == -1)

        top_vec = self.bert(src, token_type_ids=segs, attention_mask=mask_src)
        top_vec = top_vec.last_hidden_state
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        
        loss = 0
        if labels is not None:
            loss = self.loss(sent_scores, labels)
            
            loss = (loss * mask_cls.float()).sum() / len(labels)
        
        return loss, sent_scores
    
    def step(self, batch):

        src = batch['src']
        if len(batch['labels']) > 0 :
            labels = batch['labels']
        else:
            labels = None
        segs = batch['segs']
        clss = batch['clss']
        
        loss, sent_scores = self(src, segs, clss, labels)    
        
        return loss, sent_scores, labels

    def training_step(self, batch, batch_idx):

        loss, sent_scores, labels = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def validation_step(self, batch, batch_idx):
        
        loss, sent_scores, labels = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def test_step(self, batch, batch_idx):
        
        loss, sent_scores, labels = self.step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def acc_loss(self, outputs):
        total_loss = 0
        hit_cnt = 0
        for outp in outputs:
            labels = outp['labels'].cpu()
            predictions, idxs = outp['predictions'].cpu().sort()
            loss = outp['loss'].cpu()
            for label, idx in zip(labels, idxs):
                for i in range(1,3):
                    if label[idx[-i-1]] == 1 : 
                        hit_cnt += 1

            total_loss += loss
            
        avg_loss = total_loss / len(outputs)
        acc = hit_cnt / (3*len(outputs)*len(labels))
        
        return acc, avg_loss
        
    def training_epoch_end(self, outputs):
        
        acc, avg_loss = self.acc_loss(outputs)
        
        print('acc:', acc, 'avg_loss:', avg_loss)
        
        self.log('avg_train_loss', avg_loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        
        acc, avg_loss = self.acc_loss(outputs)
        
        print('val_acc:', acc, 'avg_val_loss:', avg_loss)
        
        self.log('avg_val_loss', avg_loss, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        
        acc, avg_loss = self.acc_loss(outputs)
        
        print('test_acc:', acc, 'avg_test_loss:', avg_loss)
        
        self.log('avg_test_loss', avg_loss, prog_bar=True, logger=True)

        return
        
    def configure_optimizers(self):
        
        optimizer = AdamW(self.parameters(), lr=2e-5)

        steps_per_epoch=len(train_df) // BATCH_SIZE
        total_training_steps = steps_per_epoch * N_EPOCHS
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch,
            num_training_steps=total_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

trained_model = Summarizer.load_from_checkpoint('checkpoints/best-checkpoint.ckpt') # bertsum 경로
trained_model.eval()
trained_model.freeze()

def data_process(text):
    # 문장 분리 하고,
    sents = kss.split_sentences(text)
    
    #데이터 가공하고,
    tokenlist = []
    for sent in sents:
        tokenlist.append(tokenizer(
            text = sent,
            add_special_tokens = True)) #, # Add '[CLS]' and '[SEP]'

    src = [] # 토크나이징 된 전체 문단
    labels = []  # 요약문에 해당하면 1, 아니면 0으로 문장수 만큼 생성
    segs = []  #각 토큰에 대해 홀수번째 문장이면 0, 짝수번째 문장이면 1을 매핑
    clss = []  #[CLS]토큰의 포지션값을 지정

    odd = 0

    for tkns in tokenlist:

        if odd > 1 : odd = 0
        clss = clss + [len(src)]
        src = src + tkns['input_ids']
        segs = segs + [odd] * len(tkns['input_ids'])
        odd += 1

        #truncation
        if len(src) == MAX_TOKEN_COUNT:
            break
        elif len(src) > MAX_TOKEN_COUNT:
            src = src[:MAX_TOKEN_COUNT - 1] + [src[-1]]
            segs = segs[:MAX_TOKEN_COUNT]
            break

    #padding
    if len(src) < MAX_TOKEN_COUNT:
        src = src + [0]*(MAX_TOKEN_COUNT - len(src))
        segs = segs + [0]*(MAX_TOKEN_COUNT - len(segs))

    if len(clss) < MAX_TOKEN_COUNT:
        clss = clss + [-1]*(MAX_TOKEN_COUNT - len(clss))

    return dict(
        sents = sents, #정답 출력을 위해...
        src = torch.tensor(src),
        segs = torch.tensor(segs),
        clss = torch.tensor(clss),
    )

def summarize_test(text):
    data = data_process(text.replace('\n',''))
    
    #trained_model에 넣어 결과값 반환
    _, rtn = trained_model(data['src'].unsqueeze(0), data['segs'].unsqueeze(0), data['clss'].unsqueeze(0))
    rtn = rtn.squeeze()
    
    # 예측 결과값을 받기 위한 프로세스
    rtn_sort, idx = rtn.sort(descending = True)
    
    rtn_sort = rtn_sort.tolist()
    idx = idx.tolist()

    end_idx = rtn_sort.index(0)

    rtn_sort = rtn_sort[:end_idx]
    idx = idx[:end_idx]
    
    if len(idx) > 3:
        rslt = idx[:3]
    else:
        rslt = idx
        
    summ = []
    sum_i = []
    #print(' *** 입력한 문단의 요약문은 ...')
    res = ""
    for i, r in enumerate(rslt):
        summ.append(data['sents'][r])
        sum_i.append(r)
        #print('[', i+1, ']', summ[i])

    print(sum_i)
    return summ