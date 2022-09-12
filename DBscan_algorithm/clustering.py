#!/usr/bin/python3
import argparse
import numpy as np
import os
from collections import deque,Counter
from utils import data_read,get_neighbor_sparse_matrix


def parse_arg()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_name",type=str)
    parser.add_argument("n",type=int)
    parser.add_argument("eps",type=int)
    parser.add_argument("minptr",type=int)

    return parser.parse_args()

def main(args:argparse.Namespace):
    INPUT_ROOT = "./data-2"
    INPUT_NAME = args.input_name
    INPUT_PATH = os.path.join(INPUT_ROOT,INPUT_NAME)
    OUTPUT_ROOT = "./test-2"

    N = args.n
    EPS = args.eps
    MinPtr = args.minptr

    objs,points = data_read(INPUT_PATH)
    neighbor_sparse_matrix = get_neighbor_sparse_matrix(points,EPS)

    n_neighbors = neighbor_sparse_matrix.sum(axis=1)
    core_points = n_neighbors >= MinPtr
    labels = np.full_like(objs,-1)

    dbscan(objs,labels,core_points,neighbor_sparse_matrix)

    selected_cluster = list(Counter(labels[labels!=-1]).keys())[:N]

    save_num=0
    for cluster_num in selected_cluster:
        OUTPUT_NAME = '_'.join([INPUT_NAME.split('.')[0],"cluster",str(save_num)]) +'.txt'
        OUTPUT_PATH = os.path.join(OUTPUT_ROOT,OUTPUT_NAME)

        cluster_objs = objs[(labels==cluster_num)]

        with open(OUTPUT_PATH,'w') as f:
            f.write('\n'.join(list(map(str,cluster_objs.tolist()))))

        save_num+=1

def dbscan(objs:np.array, labels:np.array, core_points:np.array, neighbor_sparse_matrix:np.array):
    cluster_num=0
    q = deque([])

    for i in range(len(objs)):
        obj = objs[i]
        if labels[obj] !=-1 or not core_points[obj]:
            continue

        while True:
            if labels[obj] == -1:
                labels[obj] = cluster_num
                if core_points[obj]:
                    neighbors = objs[neighbor_sparse_matrix[obj]]
                    q.extend(neighbors[labels[neighbors]==-1].tolist())
            
            if len(q)==0:
                break

            obj = q.popleft()

        cluster_num+=1

    return

if __name__=="__main__":
    args = parse_arg()
    main(args)