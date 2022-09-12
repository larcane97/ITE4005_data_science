#!/usr/bin/python3
from typing import List,FrozenSet
import argparse
from itertools import combinations
from utils import *
import math

def parse_arg()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('rel_min_sup',type=float)
    parser.add_argument('input_path',type=str)
    parser.add_argument('output_path',type=str)
    return parser.parse_args()

def apriori_search(init_candidate:List[FrozenSet],db:List[str],abs_min_sup:int)->Dict[FrozenSet,int]:
    """
    apriori algorithm을 적용해 frequent patterns을 반환하는 함수

    1. db를 scan하면서 candidate의 support value를 얻는다.
    2. 만약 어떠한 candidate도 db에서 발견되지 않는 경우, apriori 종료한다.
    3. support value가 minimum support보다 낮은 candidate을 제거해
            frequent-itemset을 얻는다.
    4. 각 단계에서 얻어진 frequent-itemset을 모두 저장
    5. 현재 frequent-itemset을 self-joining해 (k+1)-candidate을 generation한다.
    6. (k+1)-candidate 중 subset이 frequent하지 않은 candidate 제거한다.
    7. (k+1)-candidate이 존재하지 않는 경우, apriori 종료한다.
    8. (k+1)-candidate로 다시 1번으로 돌아간다.
    """
    k=1
    candidate = init_candidate
    freq_pattern={}
    while True:
        counter = db_scan(db,candidate)
        if not counter:
            break
        freq_set = get_frequent_set(counter,abs_min_sup)

        for freq_item in freq_set:
            freq_pattern[freq_item]=counter[freq_item]
        k+=1
        
        generated = self_joining(freq_set)
        candidate = pruning(generated,freq_set,k)
        if not candidate:
            break

    return freq_pattern

def association_rules_searching(frequent_pattern:Dict[FrozenSet,int],num_transaction:int)->List:
    """
    frequent patterns을 입력받아 association rules을 반환하는 함수
    
    1. frequent patterns 중 길이가 긴 frequent patterns부터 탐색
    2. 1~len(l)-1까지 반복하면서 해당 길이에 맞는 l의 subset인 item_set을 생성한다.
    3. l - item_set = associative_item_set으로 만든 뒤
            item_set -> associative_item_set을 association rule로 추가
    4. 위 과정을 모든 frequent set에 대해 시행
    """
    association_rules=[]
    for l in sorted(frequent_pattern,key=lambda x : len(x),reverse=True):
        for r in range(1,len(l)):
            for item_set in combinations(l,r):
                item_set = frozenset(item_set)
                associative_item_set = l-item_set
                freq_score = frequent_pattern[l]/num_transaction
                conf_score = frequent_pattern[l]/frequent_pattern[item_set]
                association_rules.append([item_set,associative_item_set,freq_score,conf_score])

    return association_rules

def main(args):
    rel_min_sup = args.rel_min_sup/100
    input_path = args.input_path
    output_path = args.output_path

    ## prepare
    with open(input_path,'r',encoding='utf-8') as f:
        db = f.readlines()
    db = list(map(preprocessing,db))
    abs_min_sup = math.ceil(len(db)*rel_min_sup)
    init_candidate = [frozenset([i]) for i in range(100)]

    ## apriori algorithm searching
    frequent_pattern = apriori_search(init_candidate=init_candidate,
                                    db=db,abs_min_sup=abs_min_sup)

    ## association rule searching
    association_rules = association_rules_searching(frequent_pattern=frequent_pattern,num_transaction=len(db))

    ## output write
    output=[]
    for out in association_rules:
        itemset = set(out[0])
        associataive_item_set = set(out[1])
        sup = round(out[2]*100,2)
        conf = round(out[3]*100,2)
        o = f"{itemset}\t{associataive_item_set}\t{sup:.2f}\t{conf:.2f}\n"
        output.append(o)
        
    with open(output_path,'w') as f:
        f.writelines(output)

if __name__=="__main__":
    args = parse_arg()
    main(args)



    
    