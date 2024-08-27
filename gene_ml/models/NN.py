import torch
from torch import nn
from .base import Model
import copy
from torch.utils.data import TensorDataset, DataLoader


class NN(nn.Module, Model):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(8,90)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(90,90)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(90,90)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(90,1)

        self.training_loss
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x
    
    def train(self,dataloader,val_dataloader,n_epochs,train_batch_size):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        patience = 1000

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        num_val_batches = len(val_dataloader)

        best_loss = float('inf')
        best_model_weights = None
        
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
                #print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            epoch_loss = epoch_loss/num_batches

            val_loss = 0.0
            for (X, y) in val_dataloader:
                pred = self(X)
                val_loss += loss_fn(pred,y).item()

            val_loss /= num_val_batches

            print("--------------------------------")
            print(f"Epoch {i+1}: loss={epoch_loss:>7f}, val_loss={val_loss:>7f}")
            print("\n")

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(self.state_dict())
                patience_curr = patience
            else:
                patience_curr -= 1
                if patience_curr == 0:
                    break

        self.load_state_dict(best_model_weights)

    def fit(self, x, y, batch_percentage=10, n_epochs=10000):
        # assumed it is already normalised
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        batch_size = int(len(x)*(batch_percentage/100))
        data_loader = DataLoader(dataset=TensorDataset(x,y), batch_size=batch_size)
        self.train(data_loader, n_epochs, batch_size)

    def predict(self,x):
        return self(x)

        