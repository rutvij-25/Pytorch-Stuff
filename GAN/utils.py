import torch

# Saving and loading models in pytorch

#make a checkpoint and save

def save_model(model,optimizer,epoch,path):

    checkpoint = {
        'model':model,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'epoch':epoch
    }

    torch.save(checkpoint,path)

#load model

def load_model(path):
    return torch.load(path)
