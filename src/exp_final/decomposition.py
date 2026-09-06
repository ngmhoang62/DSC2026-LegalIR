"""Deterministic, label-free query views for compound legal questions."""
from __future__ import annotations

import math
import re
from collections import Counter

from .relations import ascii_words

WORD_RE=re.compile(r"\w+",re.UNICODE)
STOP={"la","va","hoac","cua","cho","duoc","co","the","nao","nhung","cac","mot","theo","quy","dinh","phap","luat","ve","trong","khi","neu","thi","ra","sao"}


def inverse_document_frequency(texts):
    texts=list(texts);df=Counter()
    for text in texts:df.update(set(ascii_words(text)))
    count=max(1,len(texts))
    return {token:math.log((count+1)/(value+1))+1 for token,value in df.items()}


def query_views(text,idf,max_views=3):
    """Return up to three non-identical views, excluding the full query."""
    words=WORD_RE.findall(text);n=len(words);candidates=[]
    if n>=8:
        width=max(6,int(math.ceil(.65*n)))
        candidates.extend((" ".join(words[:width])," ".join(words[-width:])))
    scored=[]
    normalized=ascii_words(text)
    for index,(surface,token) in enumerate(zip(words,normalized)):
        if token not in STOP and len(token)>1:scored.append((idf.get(token,1.),index,surface))
    keep=min(10,max(5,int(math.ceil(n*.5))))
    chosen=sorted(scored,key=lambda row:(-row[0],row[1]))[:keep]
    if chosen:
        rare=" ".join(surface for _,_,surface in sorted(chosen,key=lambda row:row[1]))
        candidates.append("quy định về "+rare)
    result=[];seen={" ".join(words).casefold()}
    for candidate in candidates:
        key=" ".join(candidate.split()).casefold()
        if key and key not in seen:
            result.append(candidate);seen.add(key)
        if len(result)>=max_views:break
    if not result:
        result=["nội dung pháp lý: "+" ".join(words)]
    return result
