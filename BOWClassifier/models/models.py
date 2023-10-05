from torch import nn
import torch

torch.manual_seed(42)
torch.manual_seed(42)
torch.manual_seed(42)


class SimpleLinearModel(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_size_1,
                 hidden_size_2,
                 num_classes,
                 dropout):

        super(SimpleLinearModel, self).__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_size_1),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(),

            nn.Linear(in_features=hidden_size_2, out_features=num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.linear_block(x)

        return x


if __name__ == "__main__":

    net = SimpleLinearModel(input_dim=101895, hidden_size_1=256, hidden_size_2=64, num_classes=2, dropout=0.1)
    net

