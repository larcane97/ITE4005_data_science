from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
from copy import deepcopy

def softmax(idx_count:np.array)->np.array:
    "idx_count값을 입력으로 받아 softmax 함수를 통해 확률값으로 리턴해주는 함수"
    
    divisor = np.exp(idx_count).sum()
    return np.exp(idx_count)/divisor

def read_db(train_path:str)->Tuple[np.array,List,Dict[str,int],Dict[str,set]]:
    db=[]
    with open(train_path,'r') as f:
        attribute_list = f.readline().strip().split('\t')
        while True:
            data = f.readline().strip()
            if not data:
                break
            db.append(data.split('\t'))
    db = np.array(db)
    attr2idx={attr:idx for idx,attr in enumerate(attribute_list)}
    attribute_set = {attr:set(db[:,attr2idx[attr]]) for attr in attribute_list[:-1]}
    return db,attribute_list,attr2idx,attribute_set

class Node:
    def __init__(self,db:np.array,attribute_list:List[str],attr:str=None,item:str=None,isleaf:bool=False,label:str=None):
        self.childs=[]
        self.db=db
        self.attribute_list = attribute_list
        self.attr=attr
        self.item=item
        self.isleaf=isleaf
        self.label=label

    def __repr__(self):
        return f"attr:{self.attr}, item:{self.item}, isleaf:{self.isleaf}, label:{self.label}\n"

class DecisionTree:
    def __init__(self,db,attribute_list,attr2idx,attribute_set,ratio=False,pruning_thr=0.01):
        self.attr2idx = attr2idx
        self.attribute_set = attribute_set
        self.ratio= ratio
        self.pruning_thr = pruning_thr
        self.root=self.make_decision_tree(db,attribute_list)
        
    def make_decision_tree(self,db:np.array,attribute_list:List[str],item:str=None,parent:np.array=None)->Node:
        # 종료 조건 검사
        if db.size==0:
            label = self.get_label(parent.db)
            return Node(db=None,attribute_list=None,item=item,isleaf=True,label=label)
        if len(attribute_list)==0:
            label = self.get_label(db)
            return Node(db=None,attribute_list=None,item=item,isleaf=True,label=label)
        if self.get_info(db)<=self.pruning_thr:
            label = self.get_label(db)
            return Node(db=None,attribute_list=None,item=item,isleaf=True,label=label)

        attribute_list = deepcopy(attribute_list)
        node = Node(db=db,attribute_list=attribute_list,item=item)
        # 1. attribute selecting
        max_gain=-1
        best_attr=None
        for attr in np.random.choice(attribute_list,size=len(attribute_list)-1,replace=False):
            gain = self.get_information_gain(db=db,attribute=attr,ratio=self.ratio)
            if max_gain < gain:
                max_gain = gain
                best_attr = attr

        ## 2. prepruning
        if max_gain < self.pruning_thr:
            label = self.get_label(db)
            return Node(db=None,attribute_list=None,item=item,isleaf=True,label=label)

        assert best_attr is not None,f'[ERROR] Wrong gain! {len(attribute_list)}, max_gain : {max_gain}, best_attr : {best_attr}'

        node.attr=best_attr
        # 3. child node 생성 및 db 할당
        attribute_list.remove(best_attr)
        for item in self.attribute_set[best_attr]:
            sub_db = db[db[:,self.attr2idx[best_attr]]==item,:]
            node.childs.append(
                self.make_decision_tree(
                    db=sub_db,
                    attribute_list=attribute_list,
                    item=item,
                    parent=node
                    ))

        return node

    def inference(self,test_db):
        labels=[]
        for data in test_db:
            node = self.root
            while not node.isleaf:
                find=False
                attr_idx = self.attr2idx[node.attr]
                item = data[attr_idx]
                for child in node.childs:
                    if child.item==item:
                        find=True
                        node = child
                        break
                assert find,"[ERROR]not find leaf node!!"
            labels.append(node.label)
        return labels

    def get_info(self,db:np.array)->float:
        info=0
        for v in Counter(db[:,-1]).values():
            p = v/len(db)
            info -= p*np.log2(p+ 1e-9)
        return info
    
    def get_split_info(self,db:np.array,attribute:str)->float:
        split_info=0
        for item in self.attribute_set[attribute]:
            sub_db = db[db[:,self.attr2idx[attribute]]==item,:]
            p = len(sub_db)/len(db)
            split_info -= p*np.log2(p+ 1e-9)
        return split_info

    def get_attribute_info(self,db:np.array,attribute:str)->float:
        attr_info=0
        for item in self.attribute_set[attribute]:
            sub_db = db[db[:,self.attr2idx[attribute]]==item,:]
            attr_info +=(len(sub_db)/len(db))*self.get_info(sub_db)
        return attr_info

    def get_information_gain(self,db:np.array,attribute:str,ratio:bool=False)->float:
        info = self.get_info(db)
        attr_info = self.get_attribute_info(db,attribute)
        gain = info-attr_info
        if ratio:
            split_info = self.get_split_info(db,attribute)
            return gain/split_info
        return gain
       
    def get_label(self,db:np.array)->str:
        return Counter(db[:,-1]).most_common()[0][0]