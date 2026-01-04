import torch
import torch.nn as nn
import torch.nn.functional as F

class MineNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_size=100):
        super(MineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Xavier Initialization giúp hội tụ ổn định hơn
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x, y):
        input_tensor = torch.cat((x, y), dim=1)
        h1 = F.elu(self.fc1(input_tensor))
        h2 = F.elu(self.fc2(h1))
        out = self.fc3(h2)
        return out
