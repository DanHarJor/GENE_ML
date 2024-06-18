import torch
from torch import nn
from .base import Model

class NN(nn.Module, Model):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(8,50)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(50,50)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(50,1)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
    def train(self,dataloader,n_epochs,train_batch_size):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        
        for i in range(n_epochs):
            epoch_loss = 0.0

            for batch, (X, y) in enumerate(dataloader):
                pred = self(X)
                loss = loss_fn(pred,y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss, current = loss.item(), batch * train_batch_size + len(X)
                epoch_loss = epoch_loss + loss
                print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            epoch_loss = epoch_loss/num_batches
            print("--------------------------------")
            print(f"Epoch {i+1}: loss={epoch_loss:>7f}")
            print("\n")

    def predict(self,x):
        return self(x)

        