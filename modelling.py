import torch
import torch.nn.functional as F
from torch import nn, optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size = 128, num_layers=1, v_dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        dropout = v_dropout,
        bidirectional=True,
        batch_first=True)
        
        self.fc = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, 1)
        self._reinitialize()
    
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        x, _ = self.lstm(x)

        x = F.silu(self.fc(x))
        x = self.out(x)

        return x

class L1Loss_masked(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, y, u_out):

        mask = 1 - u_out
        mae = torch.abs(mask * (y - preds))
        mae = torch.sum(mae) / torch.sum(mask)

        return mae
    
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = []
    
    model.train()
    
    for X, target in dataloader:
        X, target = X.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, target) #loss_fn(output, target, X[:,:,4].reshape(-1,80,1)) for masked
        loss.backward()
        # Check for bad gradients
        """
        with torch.no_grad():
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(total_norm)
        """
        optimizer.step()
        
        train_loss.append(loss.detach().item())
           
    return train_loss
  
def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss = []
    
    model.eval()
    
    for X, target in dataloader:
        with torch.no_grad():
            X, target = X.to(device), target.to(device)

            output = model(X)
            loss = loss_fn(output, target, X[:,:,2].reshape(-1,80,1))
            
            valid_loss.append(loss.detach().item())
            
    return valid_loss