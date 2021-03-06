# -*- coding: utf-8 -*-
"""1.2 Mecab 명사 추출.ipynb
"""

import pandas as pd

from tqdm import tqdm

from konlpy.tag import Mecab

mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic/")

data = 'Y:/SeeValue/ipynb/Python/Project/BigKinds/data'

document_1 = pd.read_csv(data+"/news_20211201_20211210.csv", encoding='CP949', dtype={'news_id':str}) # 155795 
document_2 = pd.read_csv(data+"/news_20211211_20211220.csv", encoding='CP949', dtype={'news_id':str}) # 127643, 0 = 155795
document_3 = pd.read_csv(data+"/news_20211221_20211231.csv", encoding='CP949', dtype={'news_id':str}) # 155999, 0 = 283438

document = pd.concat([document_1,document_2,document_3])

document.reset_index(inplace=True, drop=True)

document['category'] = [str[4:5] for str in document.category_infra_kpf]
document['bigkinds'] = [str[16:18] for str in document.category_infra_kpf]
 
len(document), document.columns

document.groupby(document['category']).count()

pd.crosstab(index=document['category'], columns=document['bigkinds'])

"""1:정치(7) 2:경제(14) 3:사회(10) 4:문화(11) 5:국제(9) 6:지역(15) 7:스포츠(11) 8:IT_과학(6) 9:기타"""

document.head()

document.content[0]

document['content'] = document['content'].str.replace(pat=r'[^0-9A-Za-z가-힣ㄱ-ㅎ@ ]', repl=r' ', regex=True)

document.content[0]

"""## Mecab Dictionary
- Mecab 제공 사전 사용
"""

lines = document['content']

results   = []
words_all = []

print('[명사 추출 중입니다.]')
for line in tqdm(lines):
    r = []
    malist = mecab.nouns(line)
    for word in malist:
        if len(word) > 1 and word.isdigit() == False: # 두 글자 이상이면서 숫자가 아닌 경우만 저장
            r.append(word)
            words_all.append(word)
    r1 = (" ".join(r)).strip()
    results.append(r1)     
print('[명사 추출을 완료하였습니다.]\n')

print("추출된 전체 단어 수 : ", len(words_all))
print("추출된 단어 수      : ", len(set(words_all)))

vocabulary = list(set(words_all))
with open(data+"/mecab_words_all.txt", "w") as file:
    file.write('\n'.join(vocabulary))

mecab_nouns = pd.DataFrame(results, columns=['mecab_nouns'])
df          = pd.concat([document, mecab_nouns], axis=1)
df.to_csv(data+"/document_202112.csv", index=False)

"""## User Dictionary
- BigKinds 사용자 사전 사용
"""

data = 'Y:/SeeValue/ipynb/Python/Project/BigKinds/data'

document = pd.read_csv(data+"/document_202112.csv", dtype={'news_id':str})
len(document), document.columns

document.head(3)

lines = document['content']

results   = []
words_all = []

print('[명사 추출 중입니다.]')
for line in tqdm(lines):
    r = []
    malist = mecab.nouns(line)
    for word in malist:
        if len(word) > 1 and word.isdigit() == False: # 두 글자 이상이면서 숫자가 아닌 경우만 저장
            r.append(word)
            words_all.append(word)
    r1 = (" ".join(r)).strip()
    results.append(r1)     
print('[명사 추출을 완료하였습니다.]\n')

print("추출된 전체 단어 수 : ", len(words_all))
print("추출된 단어 수      : ", len(set(words_all)))

vocabulary = list(set(words_all))
with open(data+"/bigkinds_words_all.txt", "w") as file:
    file.write('\n'.join(vocabulary))

bigkinds_nouns = pd.DataFrame(results, columns=['bigkinds_nouns'])
df             = pd.concat([document, bigkinds_nouns], axis=1)
df.to_csv(data+"/document_202112.csv", index=False)

df.head(3)

df.mecab_nouns[101]

df.bigkinds_nouns[101]

