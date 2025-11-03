from torch import nn
import torch
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, in_features, mlp_ratio = 4, act_layer = nn.GELU, drop_rate = 0.):
        super().__init__()
        out_features = in_features
        hidden_features = int(in_features * mlp_ratio)
        drop_probs = (drop_rate, drop_rate)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


if __name__ == '__main__':
    mlp = MLP(in_features=512, hidden_features=1024)
    x = torch.randn(1, 512)
    y = mlp(x)
    print(y.shape)