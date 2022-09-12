import argparse
import torch
import torch.optim as optim
from model import RecommendNet
from loss import RecommendLoss
from utils import *
import os

def parse_arg()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path',type=str)
    parser.add_argument('test_path',type=str)
    parser.add_argument('--config_path',type=str,default="./config.json")
    return parser.parse_args()


def train(train_path:str,test_path:str,config_path:str):
    matrix,train_data,_ = get_data(train_path,test_path)
    config = get_config(config_path)

    if not isinstance(matrix,torch.Tensor):
        matrix = torch.Tensor(matrix)
    
    STEPS = config['STEPS']
    EPOCHS = config['EPOCHS']
    EARLY_STOP = config["EARLY_STOP"]
    log = config["log"]
    hidden_dims = config['hidden_dims']
    stop_count=0
    STOP=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RecommendNet(user_num=matrix.shape[0],item_num=matrix.shape[1],hidden_dims=hidden_dims)
    criterion = RecommendLoss()
    optimizer = optim.AdamW(model.parameters(),lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=1e-1,patience=STEPS,verbose=True,min_lr=1e-5)
    
    model = model.to(device)

    best_loss=1e+9
    matrix_list=[]
    for epoch in range(EPOCHS):
        if STOP:
            print("Early stoped..!!")
            break
        print("="*100)
        print("TRAINING..")
        model.train()
        for step in range(STEPS):
            data = matrix.clone().to(device)

            optimizer.zero_grad()

            pred= model.forward(data)
            loss = criterion(data,pred)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if best_loss > loss.detach().cpu().item():
                best_loss = loss.detach().cpu().item()
                stop_count=0
            else:
                stop_count+=1
                if stop_count>=EARLY_STOP:
                    STOP=True
                    break

            if (step+1)%log==0:
                print(f"[epoch:{epoch+1}, step:{step+1}] loss : {loss.detach().item() : .3f}")

        print(f"FINISH {epoch+1} epoch.")


        print("="*100)
        print("EVALUATION..")
        with torch.no_grad():
            model.eval()
            pred = model.forward(matrix.to(device))
            pred = pred.clip(min=1,max=5)
            matrix_list.append(pred.detach().cpu().numpy())

            print(f"UPDATE matrix..")
            matrix = pred

            for idx,d in enumerate(train_data[:,[0,1]]):
                matrix[d[0],d[1]]=train_data[idx,2]

            matrix = matrix.clip(min=1,max=5)  

    model_weights = os.path.basename(train_path).split('.')[0] + ".pt"
    with open(model_weights,'wb') as f:
        torch.save({'model':model.state_dict(),'matrix':matrix},f)
    print("save model weights in {}.".format(model_weights))

if __name__=="__main__":
    args = parse_arg()
    train(args.train_path,args.test_path,args.config_path)


