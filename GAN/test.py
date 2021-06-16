import torch.nn as nn
import torch

# Testing the crossentropy loss in pytorch

target = torch.LongTensor([1]) #pytorch converts it into [0,1]

output = torch.tensor([[0.2,0.8]]) #pytorch takes softmax of this => [0.355,0.645]

loss = nn.CrossEntropyLoss()

print(loss(output,target)) # -1 * ln(0.645)

# Testing the binary crossentropy loss in pytorch

target = torch.Tensor([1,0])

output = torch.Tensor([0.2,0.8])

loss = nn.BCELoss()

print(loss(output,target))
