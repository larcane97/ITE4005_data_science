from typing import List,Dict,FrozenSet
from itertools import combinations, permutations

def preprocessing(transaction:str)->FrozenSet:
    """
    하나의 string transaction 데이터를 입력으로 받아 int format의 set으로 반환해주는 함수

    Parameters:
        transaction(str) : db의 하나의 trasaction, str으로 표현되어 있다.

    Returns:
        FrozenSet[int]
    """
    transaction = transaction.strip()
    transaction = transaction.split('\t')
    transaction = list(map(int,transaction))
    return frozenset(transaction)

def db_scan(db:List[str],candidate:List[FrozenSet])->Dict[FrozenSet,int]:
    """
    전체 db를 scan하면서 candidate의 횟수를 카운팅하는 함수

    Parameters:
        db(List[str]) : 전체 db list
        candidate(List[FrozenSet]) : candidate list

    Returns:
        Dict[FrozenSet,int] : candidate을 key값으로, frequency를 values값으로 갖는 dict
    """
    counter={}
    for transaction in db:
        for c in candidate:
            if c.issubset(transaction):
                if counter.get(c) is None:
                    counter[c]=1
                else:
                    counter[c]+=1
    return counter

def get_frequent_set(counter:Dict[FrozenSet,int],abs_min_sup:int)->List[FrozenSet]:
    """
    각 candidate의 frequency와 absolute minimum support value를
    입력으로 받아 frequent itemset을 반환하는 함수.

    Parameters:
        counter(Dict[FrozenSet,int]) : candidate을 key값으로, frequency를 values값으로 갖는 dict
        abs_min_sup(int) : absolute minimum support value

    Returns:
        List[FrozenSet] : frequent itemset
    """
    freq_set=list()
    for itemset,freq in counter.items():
        if freq>=abs_min_sup:
            freq_set.append(itemset)
    return freq_set

def self_joining(freq_set:List[FrozenSet])->List[FrozenSet]:
    """
    (k-1)-itemset을 입력으로 받아 self-joining으로 k-candidate을 생성(generation)

    Parameters:
        freq_set(List[FrozenSet]) : (k-1)-itemset

    Returns:
        List[FrozenSet] : k-candidate
    """    
    generated=list()
    for L1,L2 in permutations(freq_set,2):
        joinable=True
        for i,(item1,item2) in enumerate(zip(sorted(L1),sorted(L2))):
            if i==len(L1)-1:
                if item1>=item2:
                    joinable=False
                break
            if item1!=item2:
                joinable=False
                break
            
        if not joinable:
            continue
        r = (L1|L2)
        generated.append(r)
    return generated

def pruning(generated:List[FrozenSet],prev_freq:List[FrozenSet],k:int)->List[FrozenSet]:
    """
    k-candidate의 subset이 infrequent한 경우 candidate에서 제외하는 함수.

    Parameters:
        generated(List[FrozenSet]) : k-candidate
        prev_freq(List[FrozenSet]) : (k-1)-itemset
        k(int) : k

    Returns:
        List[FrozenSet] : subset이 infrequent한 candidate을 제외한 set
    """
    result=list()
    for candidate in generated:
        isCandidate=True
        for subset in combinations(candidate,k-1):
            subset = frozenset(subset)
            if subset not in prev_freq:
                isCandidate=False
                break
        if isCandidate:
            result.append(candidate)
    return result