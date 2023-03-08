import numpy as np
import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from bareunpy import Tagger

API_KEY = "koba-M3CMSHQ-ZRBUUNI-WHTY2KI-NUYJDCY" # KPF에서 제공하는 API_KEY
tagger = Tagger(API_KEY,'175.106.88.249',5757) # KPF에서 제공하는 바른 형태소분석기

model = SentenceTransformer('kpfSBERT')

def keyword_ext(text):

    tokenized_doc = tagger.pos(text)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'NNG' or word[1] == 'NNP'])

    n_gram_range = (1,1)

    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names_out()

    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)

    return mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2)

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    print(keywords_idx)

    return [words[idx] for idx in keywords_idx]

if __name__ == "__main__":
    text = """
    한덕수 국무총리는 30일 난방비 인상 등과 관련해 시장 상황에 맞게 가격을 조정하지 않는 정책은 '포퓰리즘'이라고 규정했다.

    한 총리는 이날 오전 정부서울청사에서 주재한 국무회의 모두발언에서 "한파와 가스비 등 공공요금 인상이 겹쳐 국민들이 느끼는 고통에 마음이 무겁다"고 말했다.
    
    이어 "그러나 국민들이 불편해한다고 해서 장기간 조정해야 할 가격을 시장에 맞서 조정하지 않고 억누르는 정책은, 추후 국민들께 더 큰 부담을 드리고 우리 경제에 악영향을 끼치는 포퓰리즘 정책에 다름 아니다"라고 지적했다.
    
    그러면서 "이번 난방비 문제를 통해 이를 다시 한번 확인하는 계기가 됐다고 생각한다"고 덧붙였다.
    
    직접 거론하지는 않았지만, 전임 문재인 정부가 최근 몇 년간 난방비 인상 요인이 있었음에도 가격을 올리지 않아 이번 '난방비 폭탄' 사태가 초래됐다는 시각을 우회적으로 드러낸 것으로 보인다.
    
    이날 국무회의에서 의결되는 취약계층 에너지바우처 지원 확대와 관련해서는 "기획재정부·산업통상자원부 등 관계부처가 신속한 집행과 함께 혜택을 몰라서 지원받지 못하는 분이 없도록 제도 안내와 홍보에도 각별히 노력해달라"고 당부했다.
    
    한 총리는 "올해는 작년도 물가 인상에 대응하기 위해 우리나라를 포함해 많은 나라가 급속히 추진할 수밖에 없었던 금리 인상의 충격이 나타나는 한 해가 될 것"이라며 "우리 경제도 매우 힘들고 어려운 한 해가 될 것으로 예상한다"고 말했다.
    
    그러면서 "이런 때일수록 조정 과정에서 어려움을 겪을 수밖에 없는 사회적 약자를 위해 최선의 보완책을 마련하는 데 온 힘을 기울여야 한다"고 했다.
    
    특히 국가 운영을 위해 불가피하게 공공요금을 인상하는 경우에는 국민이 취할 수 있는 대안에 대한 정보도 충분히 제공하라고 관계 부처에 거듭 당부했다.
    
    한 총리는 이날부터 실내 마스크 착용 의무가 2020년 11월 이후 약 2년 3개월 만에 해제된다면서 "그간 불편을 감내하며 마스크 착용에 협조해준 국민 여러분께 깊이 감사드린다"고 말했다.
    
    이어 "아직 코로나의 위험에서 완전히 벗어난 것은 아니다"라며 의료기관·대중교통 등에서는 마스크 착용 의무가 당분간 유지된다고 강조했다.
    
    특히 어르신 등 고위험군은 위험하다고 판단되는 장소에서는 언제든지 마스크를 착용해야 한다고 덧붙였다.
    
    한 총리는 "정부는 국민 모두가 코로나로부터 좀더 자유로워질 수 있도록 더욱 노력하겠다"며 백신 접종과 일상 속 방역 수칙 실천에도 동참해달라고 당부했다.
    """

    print(keyword_ext(text))
