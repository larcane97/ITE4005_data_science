#!/usr/bin/python3
import numpy as np
import argparse
from src import DecisionTree,read_db,softmax

def parse_arg()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path',type=str)
    parser.add_argument('test_path',type=str)
    parser.add_argument('result_path',type=str)
    parser.add_argument('--tree_num',type=int,default=500)
    parser.add_argument('--pruning_thr',type=float,default=1e-3)
    return parser.parse_args()

def main(args):
    TRAIN_TXT=args.train_path
    TEST_TXT=args.test_path
    RESULT_TXT=args.result_path
    tree_num = args.tree_num
    pruning_thr = args.pruning_thr

    db,attribute_list,attr2idx,attribute_set = read_db(TRAIN_TXT)
    test_db,_,_,_ = read_db(TEST_TXT)

    label_category = np.unique(db[:,-1])
    cat2num = {label:num for num,label in enumerate(label_category)}
    num2cat = {num:label for num,label in enumerate(label_category)}
    cat2num_vec = np.vectorize(lambda x : cat2num[x])
    num2cat_vec = np.vectorize(lambda x : num2cat[x])

    total_idx = np.arange(db.shape[0])
    idx_count = np.zeros_like(total_idx)

    forest=[]
    for _ in range(tree_num):
        # 1. data sampling with replacement
        train_idx = np.random.choice(total_idx,size=db.shape[0]*2,replace=True,p=softmax(idx_count))
        val_idx = np.setdiff1d(total_idx,train_idx)
        train_db = db[train_idx]
        val_db = db[val_idx]
        
        # 2. tree creation
        tree = DecisionTree(db=train_db,attribute_list=attribute_list[:-1],attribute_set=attribute_set,attr2idx=attr2idx,ratio=True,pruning_thr=pruning_thr)

        # 3. inference
        label = val_db[:,-1]
        pred = np.array(tree.inference(val_db[:,:-1]))
        
        # 4. counting
        wrong_idx = val_idx[label!=pred]
        idx_count[wrong_idx]+=1

        acc = (label==pred).sum()/label.shape[0]
        forest.append([tree,acc])

    preds = [tree[0].inference(test_db) for tree in forest]
    preds = np.array(preds)

    # 5. weighted summation
    logits=None
    for i in range(len(preds)):
        if logits is None:
            logits = np.eye(len(label_category))[cat2num_vec(preds[i])] * forest[i][1]
        else:
            logits += np.eye(len(label_category))[cat2num_vec(preds[i])] * forest[i][1]
    logits /=len(preds)

    labels = num2cat_vec(logits.argmax(axis=1))
    
    dt_result=['\t'.join(attribute_list) + "\n"]
    for data,label in zip(test_db.tolist(),labels):
        out = data + [label]
        out = '\t'.join(out) + "\n"
        dt_result.append(out)
    with open(RESULT_TXT,'w') as f:
        f.writelines(dt_result)
        
if __name__=="__main__":
    args = parse_arg()
    main(args)