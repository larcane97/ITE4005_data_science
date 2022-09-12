#!/usr/bin/python3
import argparse
from model import RecommendNet
from train import *
import os
from utils import *
import torch

def parse_arg()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path',type=str)
    parser.add_argument('test_path',type=str)
    parser.add_argument('--config_path',type=str,default="./config.json")
    return parser.parse_args()

def evaluation(train_path,test_path,config_path):
    model_weights = os.path.basename(train_path).split('.')[0] + ".pt"
    if not os.path.exists(model_weights):
        print("model weights don't exist.")
        print("start train model")
        train(train_path,test_path,config_path)
        print("end train model")
    
    print("evaluation..")
    _,_,test_data = get_data(train_path,test_path)
    config = get_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_weights,map_location=device)
    matrix = checkpoint['matrix']

    model = RecommendNet(user_num=matrix.shape[0],item_num=matrix.shape[1],hidden_dims=config['hidden_dims'])
    model.load_state_dict(checkpoint['model'])
    

    model = model.to(device)
    matrix = matrix.to(device)
    with torch.no_grad():
        model.eval()
        pred = model.forward(matrix)

    pred_list=[]
    for t in test_data:
      pred_list.append(f"{t[0]}\t{t[1]}\t{pred[t[0],t[1]]}\n")

    save_path = os.path.basename(train_path).split('.')[0]+".base_prediction.txt"
    with open(save_path,"w") as f:
        f.writelines(pred_list)
    print("save output file in {}.".format(save_path))

if __name__=="__main__":
    args = parse_arg()
    evaluation(args.train_path,args.test_path,args.config_path)
    

