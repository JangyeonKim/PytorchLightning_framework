import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self) :
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
        
    def forward(self, x) :
        return self.model(x)
    
def MainModel(**kwargs) :
    model = NeuralNetwork(**kwargs)
    return model    