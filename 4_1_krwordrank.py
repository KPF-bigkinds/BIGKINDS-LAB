# -*- coding: utf-8 -*-
"""4.1 krwordrank.ipynb
"""

import pandas as pd

corona = pd.read_csv('Y:/SeeValue/ipynb/data/krwordrank.csv', encoding='CP949', dtype={'news_id':'str'}

! pip install KRWordRank
from krwordrank.word     import KRWordRank
from krwordrank.sentence import summarize_with_sentences

stopwords    = []
keywords_lst = []
sents_lst    = []

for sentences in corona.content:
    keywords, sents = summarize_with_sentences(list(filter(bool, sentences.split('.'))),
                                               stopwords    = stopwords, 
                                               num_keywords = 5, 
                                               num_keysents = 3)
    keywords_lst.append(keywords)
    sents_lst.append(sents)
    
df_sents = pd.DataFrame(sents_lst)

tmp_list = []
for dct in keywords_lst:
    tmp = ''
    for tup in dct.items():  
        tmp = tmp + tup[0] + '(' + str(tup[1]) + ') '
    tmp_list.append(tmp)
df_keywords = pd.DataFrame(tmp_list, columns=['keywords'])




