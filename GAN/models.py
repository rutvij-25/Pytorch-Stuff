import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,im_dim,z_dim):
        super(Generator,self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim,128),
            nn.LeakyReLU(0.01),
            nn.Linear(128,512),
            nn.LeakyReLU(0.01),
            nn.Linear(512,im_dim*im_dim),
            nn.Tanh()
        )
    def forward(self,x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self,im_dim):
        super(Discriminator,self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(im_dim*im_dim,128),
            nn.Linear(128,64),
            nn.LeakyReLU(0.01),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.discriminator(x)


