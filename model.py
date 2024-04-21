import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, device="cuda"):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, time):
        device = self.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TwiceConv(nn.Module):
  def __init__(self,input_channels, output_channels, time_emb_dim):
    super(TwiceConv, self).__init__()
    self.time_mlp =  None
    if time_emb_dim != -1:
        self.time_mlp = nn.Linear(time_emb_dim, output_channels)
    self.conv_pair_1 = nn.Sequential(
      nn.Conv2d(input_channels,output_channels,3,1,1,bias=False),
      nn.BatchNorm2d(output_channels),
      nn.ReLU(inplace = True),
    )
    self.relu = nn.ReLU()
    self.conv_pair_2 = nn.Sequential(
        nn.Conv2d(output_channels,output_channels,3,1,1,bias=False),
        nn.ReLU(inplace = True)
    )
  def forward(self, x, t):
    h = self.conv_pair_1(x)
    if self.time_mlp is not None:
        time_emb = self.relu(self.time_mlp(t))
        # time_emb = time_emb[(..., ) + (None, ) * 2]
        time_emb = torch.unsqueeze(time_emb, 2)
        time_emb = torch.unsqueeze(time_emb, 2)
        h = h + time_emb
    out = self.conv_pair_2(h)
    return out


class UnetArchitecture(nn.Module):
  def __init__(self, input_channels=3, output_channels=3, feature_list=[64,128,256,512], time_emb_dim=64):
    super(UnetArchitecture,self).__init__()
    self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
    self.encoder_list = nn.ModuleList()
    self.decoder_list = nn.ModuleList()
    self.pool = nn.MaxPool2d(2,2)
    self.time_emb_dim = time_emb_dim
    # encoder part  OR Down Part
    len_feature_list = len(feature_list)
    for idx in range(len_feature_list):
      self.encoder_list.append( TwiceConv(input_channels, feature_list[idx], self.time_emb_dim))
      input_channels = feature_list[idx]

    # decoder part OR Up Part
    for idx in range(1,len_feature_list+1): # reading feature_list in reverse order
      self.decoder_list.append(
              nn.ConvTranspose2d(
               feature_list[len_feature_list-idx]*2,feature_list[len_feature_list-idx],kernel_size=2,stride=2
               )
          )
      self.decoder_list.append(
             TwiceConv(feature_list[len_feature_list-idx]*2, feature_list[len_feature_list-idx], self.time_emb_dim)
          )

    self.bridge = TwiceConv(feature_list[-1], feature_list[-1]*2, self.time_emb_dim)
    self.decoder_last_conv = nn.Conv2d(feature_list[0],output_channels,kernel_size=1)

  def forward(self, x, timestep):
    skipconnection_list = []
    t = self.time_mlp(timestep)
    for elem in self.encoder_list:
      x = elem(x, t)
      skipconnection_list.append(x)
      x = self.pool(x)
    x = self.bridge(x, t)
    skipconnection_list = skipconnection_list[::-1]

    for idx in range(0, len(self.decoder_list), 2):  ## our decoder_list was appended twice !!
      x = self.decoder_list[idx](x)
      skipconnection = skipconnection_list[idx//2]
      if x.shape != skipconnection.shape:
          x = TF.resize(x, size=skipconnection.shape[2:])
      concat_skipconnection = torch.cat((skipconnection, x),dim=1)
      x = self.decoder_list[idx+1](concat_skipconnection, t)
    return self.decoder_last_conv(x)

def test():
    BATCH_SIZE = 1
    T = 300
    x = torch.randn(1,3,160,160)
    t = torch.randint(0, T, (BATCH_SIZE,))
    model = UnetArchitecture(input_channels=3,output_channels=3)
    preds=model(x,t)
    print("pred output: ",preds.shape)
    print("input shape: ",x.shape)
