import numpy as np
from typing import Tuple,Union
from sklearn.metrics import pairwise_distances

def data_preprocessing(data:str)->str:
    data = data.strip('\n').split('\t')
    data[0],data[1],data[2]=int(data[0]),float(data[1]),float(data[2])
    return data

def data_read(path:str)->Tuple[np.array,np.array]:
    data_list=[]
    obj_list=[]
    with open(path,'r') as f:
        while(True):
            data = f.readline()
            if data=='':
                break
            data = data_preprocessing(data)
            data_list.append(data[1:])
            obj_list.append(data[0])
    return np.array(obj_list,dtype=np.int32),np.array(data_list,dtype=np.float64)

def get_neighbor_sparse_matrix(data:np.array,eps:Union[int,float])->np.array:
    return pairwise_distances(data,data)<eps