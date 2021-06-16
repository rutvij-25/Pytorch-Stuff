
from torch.utils.data import DataLoader
import torchvision,torch
from torchvision import transforms
import torch.optim as optimizers
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from utils import save_model,load_model
from models import *

T = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
dataset = torchvision.datasets.MNIST(root='mnist',train=True,transform=T)

writer = SummaryWriter('writer')
step = 0
dataloader = DataLoader(dataset,batch_size=64)


#HYPERPARAMETERS
EPOCHS = 50
LEARNING_RATE = 3e-4
IMG_SIZE = 28
Z_DIM = 64
BATCH_SIZE = 64
FIXED_NOISE = torch.rand(BATCH_SIZE,Z_DIM)

G = Generator(IMG_SIZE,Z_DIM)
D = Discriminator(IMG_SIZE)

optG = optimizers.Adam(G.parameters(),LEARNING_RATE)
optD = optimizers.Adam(D.parameters(),LEARNING_RATE)

criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    for idx,(real,_) in enumerate(dataloader):
        
        batch_size = real.shape[0]
        noise = torch.rand(batch_size,Z_DIM)
        fake = G(noise)
        fake = fake.reshape(-1,IMG_SIZE*IMG_SIZE)
        real = real.reshape(batch_size,IMG_SIZE*IMG_SIZE)
        #Train Discriminator
        
        D_out_fake = D(fake).reshape(-1)
        D_out_real = D(real).reshape(-1)
        
        real_label = torch.ones_like(D_out_real)
        fake_label = torch.zeros_like(D_out_fake)

        lossD_real = criterion(D_out_real,real_label)
        lossD_fake = criterion(D_out_fake,fake_label)

        lossD = (lossD_fake + lossD_real)/2
        
        optD.zero_grad()

        lossD.backward(retain_graph=True)

        optD.step()

        #Train Generator

        D_pred = D(fake).reshape(-1)

        lossG = criterion(D_pred,torch.ones_like(D_pred))

        optG.zero_grad()

        lossG.backward()
        
        optG.step()
    
    if(epoch%5==0):
        with torch.no_grad():
            images = G(FIXED_NOISE).reshape(-1,1,IMG_SIZE,IMG_SIZE)
            grid = make_grid(images,normalize=True)
            writer.add_image('Generated',grid,step)
            writer.add_scalar('Dloss',lossD,step)
            writer.add_scalar('Gloss',lossG,step)
        save_model(G,optG,epoch,f'saved/checkpoint{epoch}.pth')
        step+=1




    