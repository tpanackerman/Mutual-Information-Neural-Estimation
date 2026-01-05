import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMineNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_size=16): 
        super(SimpleMineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x, y):
        input_tensor = torch.cat((x, y), dim=1)
        h1 = F.elu(self.fc1(input_tensor))
        out = self.output(h1)
        return out

class ComplexMineNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_size=128):
        super(ComplexMineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x, y):
        input_tensor = torch.cat((x, y), dim=1)
        h1 = F.elu(self.fc1(input_tensor))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))
        h4 = F.elu(self.fc4(h3))
        out = self.output(h4)
        return out
