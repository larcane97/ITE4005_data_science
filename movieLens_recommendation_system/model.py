import torch.nn as nn

class ResBlock(nn.Module):
  def __init__(self,in_dim,out_dim):
    super(ResBlock,self).__init__()
    self.block1= nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.GELU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_dim,in_dim),
        )
    self.block2=nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.GELU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_dim,out_dim),
        )
    
  def forward(self,x):
    return self.block2(x+self.block1(x))

class UserEmbedder(nn.Module):
  def __init__(self,item_num,hidden_dims=[256,128,64,128,256]):
    super(UserEmbedder,self).__init__()

    block_list = [ResBlock(item_num,hidden_dims[0])]
    for h_idx in range(len(hidden_dims)):
      if h_idx==0:
        continue
      block_list.append(ResBlock(hidden_dims[h_idx-1],hidden_dims[h_idx]))

    self.model=nn.Sequential(*block_list)
  
  def forward(self,x):
    # x = (user_num,item_num)
    return self.model(x)

class ItemEmbedder(nn.Module):
  def __init__(self,user_num,hidden_dims=[256,128,64,128,256]):
    super(ItemEmbedder,self).__init__()
    
    block_list = [ResBlock(user_num,hidden_dims[0])]
    for h_idx in range(len(hidden_dims)):
      if h_idx==0:
        continue
      block_list.append(ResBlock(hidden_dims[h_idx-1],hidden_dims[h_idx]))

    self.model=nn.Sequential(*block_list)
  
  def forward(self,x):
    # x = (user_num,item_num)
    return self.model(x)


class RecommendNet(nn.Module):
  def __init__(self,user_num,item_num,hidden_dims=[256,128,64,128,256]):
    super(RecommendNet,self).__init__()

    self.user_net= UserEmbedder(item_num=item_num,hidden_dims=hidden_dims)
    self.item_net= ItemEmbedder(user_num=user_num,hidden_dims=hidden_dims)

    self.init()
  
  def forward(self,x):
    # x = (user_num,item_num)
    user_vec = self.user_net(x) # (user_num,hidden_dim)
    item_vec = self.item_net(x.T) # (item_num,hidden_dim)

    result = user_vec @ item_vec.T
    
    return result # (user_num,item_num)

  def init(self):
    for m in self.modules():
      if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm1d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)