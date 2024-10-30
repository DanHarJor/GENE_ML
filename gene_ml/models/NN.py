import torch
from torch import nn
from .base import Model
import copy
from torch.utils.data import TensorDataset, DataLoader


class NN(nn.Module, Model):
    def __init__(self, n_inputs, n_layers, n_neurons_per_layer):
        self.n_layers = n_layers
        super(NN, self).__init__()
        self.linear_first = nn.Linear(n_inputs,n_neurons_per_layer)
        self.relu_first = nn.ReLU()

        self.hidden = nn.Linear(n_neurons_per_layer,n_neurons_per_layer)
        self.relu_hidden = nn.ReLU()
        self.linear_last = nn.Linear(n_neurons_per_layer,1)
        # self.relu_last = nn.ReLU()
        self.training_loss = None
    
    def forward(self,x):
        x = self.linear_first(x)
        x = self.relu_first(x)
        for i in range(self.n_layers):
            x = self.hidden(x)
            x = self.relu_hidden(x)
        x = self.linear_last(x)
        # x = self.relu_last(x)
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

    def fit(self, x, y, x_val, y_val, batch_percentage=10, n_epochs=10000):
        # assumed it is already normalised
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x_val = torch.Tensor(x_val)
        y_val = torch.Tensor(y_val)
        batch_size = int(len(x)*(batch_percentage/100))
        batch_size_val = int(len(x_val)*(batch_percentage/100))
        data_loader = DataLoader(dataset=TensorDataset(x,y), batch_size=batch_size)
        val_data_loader = DataLoader(dataset=TensorDataset(x_val,y_val), batch_size=batch_size_val)
        self.train(data_loader, val_data_loader, n_epochs, train_batch_size=batch_size)

    def predict(self,x):
        x = torch.Tensor(x)
        pred = self(x).detach().numpy()
        return pred

        