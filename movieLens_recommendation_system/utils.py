import numpy as np
import json

def get_data(train_path:str,test_path:str):
    with open(train_path,'r') as f:
        raw_data = f.readlines()
        train_data=[]
        for r in raw_data:
            r = r.strip().split("\t")
            r = list(map(int,r))
            train_data.append(r)
        train_data = np.array(train_data)    

    with open(test_path,'r') as f:
        raw_test_data = f.readlines()
        test_data=[]
        for r in raw_test_data:
            r = r.strip().split("\t")
            r = list(map(int,r))
            test_data.append(r)
        test_data = np.array(test_data)   

    user_num = max(max(train_data[:,0]),max(test_data[:,0]))
    movie_num = max(max(train_data[:,1]),max(test_data[:,1]))

    matrix = np.ones((user_num+1,movie_num+1))
    for idx,d in enumerate(train_data[:,[0,1]]):
        matrix[d[0],d[1]]=train_data[idx,2]
    
    return matrix,train_data,test_data

def get_config(config_path):
    with open(config_path,'r') as f:
        config = json.load(f)
    return config
