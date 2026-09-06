"""Set-level feature primitives for fifth-slot replacement."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .relations import amendment_relation, ascii_words, authority, is_amendment

STOP = {"la","va","hoac","cua","cho","duoc","co","the","nao","nhung","cac","mot","theo","quy","dinh","phap","luat","ve","trong","khi","neu"}


class FoldLabelStats:
    def __init__(self, labels, qids):
        self.frequency=Counter();self.gold_size_sum=Counter();self.cooccurrence=Counter()
        for qid in qids:
            gold=sorted(labels[qid]);size=len(gold)
            for doc in gold:
                self.frequency[doc]+=1;self.gold_size_sum[doc]+=size
            for left_index,left in enumerate(gold):
                for right in gold[left_index+1:]:self.cooccurrence[(left,right)]+=1

    @staticmethod
    def _key(left,right):return tuple(sorted((left,right)))

    def frequency_for(self,doc,query_gold=None):
        return self.frequency[doc]-int(query_gold is not None and doc in query_gold)

    def cooccur(self,left,right,query_gold=None):
        if left==right:return self.frequency_for(left,query_gold)
        value=self.cooccurrence[self._key(left,right)]
        if query_gold is not None and left in query_gold and right in query_gold:value-=1
        return value

    def mean_gold_size(self,doc,query_gold=None):
        count=self.frequency_for(doc,query_gold)
        total=self.gold_size_sum[doc]
        if query_gold is not None and doc in query_gold:total-=len(query_gold)
        return total/max(1,count)


def token_set(text):return {word for word in ascii_words(text) if word not in STOP and len(word)>1}
def jaccard(left,right):return len(left&right)/max(1,len(left|right))


def slate_features(question,ranking,candidate_index,systems,titles,stats,query_gold=None):
    """Features for replacing rank five with one candidate at index 5..9."""
    if candidate_index<5 or candidate_index>=len(ranking):raise ValueError("Candidate must be below the top-five boundary")
    candidate,displaced=ranking[candidate_index],ranking[4];top4=ranking[:4]
    qtokens=token_set(question);ctokens=token_set(titles.get(candidate,""));dtokens=token_set(titles.get(displaced,""))
    top_tokens=[token_set(titles.get(doc,"")) for doc in top4];covered=set().union(*top_tokens) if top_tokens else set()
    value=[float(candidate_index+1),float(candidate_index-4)]
    candidate_ranks=[];displaced_ranks=[]
    for order in systems.values():
        rank={doc:i for i,doc in enumerate(order,1)}
        cr=float(rank.get(candidate,65));dr=float(rank.get(displaced,65));candidate_ranks.append(cr);displaced_ranks.append(dr)
        value.extend([cr,dr,cr-dr,1/(1+cr)-1/(1+dr),float(cr<=5)-float(dr<=5)])
    value.extend([
        min(candidate_ranks),min(displaced_ranks),np.mean(candidate_ranks)-np.mean(displaced_ranks),
        np.std(candidate_ranks)-np.std(displaced_ranks),
        sum(rank<=5 for rank in candidate_ranks)-sum(rank<=5 for rank in displaced_ranks),
    ])
    candidate_overlap=len(qtokens&ctokens)/max(1,len(qtokens));displaced_overlap=len(qtokens&dtokens)/max(1,len(qtokens))
    candidate_novel=len((qtokens&ctokens)-covered)/max(1,len(qtokens));displaced_novel=len((qtokens&dtokens)-covered)/max(1,len(qtokens))
    cj=[jaccard(ctokens,tokens) for tokens in top_tokens];dj=[jaccard(dtokens,tokens) for tokens in top_tokens]
    candidate_relations=[amendment_relation(titles.get(candidate,""),titles.get(doc,"")) for doc in top4]
    displaced_relations=[amendment_relation(titles.get(displaced,""),titles.get(doc,"")) for doc in top4]
    ca,da=authority(titles.get(candidate,"")),authority(titles.get(displaced,""));top_authority=authority(titles.get(top4[0],"")) if top4 else "other"
    query_words=ascii_words(question)
    value.extend([
        len(qtokens),sum(word in {"va","hoac","dong","thoi"} for word in query_words),sum(word.isdigit() for word in query_words),
        candidate_overlap,displaced_overlap,candidate_overlap-displaced_overlap,
        candidate_novel,displaced_novel,candidate_novel-displaced_novel,
        max(cj,default=0.),np.mean(cj) if cj else 0.,max(dj,default=0.),np.mean(dj) if dj else 0.,
        float(is_amendment(titles.get(candidate,""))),float(is_amendment(titles.get(displaced,""))),
        float(any(flag for flag,_ in candidate_relations)),float(any(reason=="instrument" for flag,reason in candidate_relations if flag)),
        float(any(flag for flag,_ in displaced_relations)),float(ca==top_authority),float(da==top_authority),float(ca==da),
    ])
    cf=stats.frequency_for(candidate,query_gold);df=stats.frequency_for(displaced,query_gold)
    cc=[stats.cooccur(candidate,doc,query_gold) for doc in top4];dc=[stats.cooccur(displaced,doc,query_gold) for doc in top4]
    value.extend([
        math.log1p(max(0,cf)),math.log1p(max(0,df)),math.log1p(max(0,cf))-math.log1p(max(0,df)),
        stats.mean_gold_size(candidate,query_gold),stats.mean_gold_size(displaced,query_gold),
        math.log1p(max(cc,default=0)),math.log1p(max(dc,default=0)),
        math.log1p(sum(cc)),math.log1p(sum(dc)),math.log1p(sum(cc))-math.log1p(sum(dc)),
    ])
    result=np.asarray(value,dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError("Non-finite slate feature")
    return result


def replacement_class(gold,displaced,candidate):
    candidate_gold=candidate in gold;displaced_gold=displaced in gold
    if candidate_gold and not displaced_gold:return 2
    if displaced_gold and not candidate_gold:return 0
    return 1


def dense_relation_features(query_vector, document_vectors, candidate_index):
    """Return query-conditioned content novelty for a rank-6..10 candidate."""
    docs=np.asarray(document_vectors,dtype=np.float32)
    query=np.asarray(query_vector,dtype=np.float32)
    if docs.ndim!=2 or len(docs)<6 or candidate_index<5 or candidate_index>=len(docs):
        raise ValueError("Invalid dense slate geometry")
    if docs.shape[1]!=len(query):raise ValueError("Dense dimension mismatch")
    query=query/max(float(np.linalg.norm(query)),1e-12)
    docs=docs/np.maximum(np.linalg.norm(docs,axis=1,keepdims=True),1e-12)
    top4=docs[:4];candidate=docs[candidate_index];displaced=docs[4]
    candidate_similarity=top4@candidate;displaced_similarity=top4@displaced
    centroid=top4.mean(0);centroid/=max(float(np.linalg.norm(centroid)),1e-12)
    gram=top4@top4.T+np.eye(4,dtype=np.float32)*1e-4
    coefficient=np.linalg.solve(gram,top4@query)
    residual=query-coefficient@top4
    residual_norm=float(np.linalg.norm(residual));residual_unit=residual/max(residual_norm,1e-12)
    cq=float(candidate@query);dq=float(displaced@query)
    values=[
        cq,dq,cq-dq,float(candidate@displaced),
        float(candidate@centroid),float(displaced@centroid),float(candidate@centroid-displaced@centroid),
        float(candidate_similarity.max()),float(candidate_similarity.mean()),float(candidate_similarity.min()),float(candidate_similarity.std()),
        float(displaced_similarity.max()),float(displaced_similarity.mean()),float(displaced_similarity.min()),float(displaced_similarity.std()),
        float(candidate_similarity.max()-displaced_similarity.max()),float(candidate_similarity.mean()-displaced_similarity.mean()),
        residual_norm,float(candidate@residual_unit),float(displaced@residual_unit),float(candidate@residual_unit-displaced@residual_unit),
    ]
    result=np.asarray(values,dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError("Non-finite dense slate feature")
    return result
