import torch
import torch.nn as nn
import torch.optim as optim


# Define model (feed-forward, two hidden layers)
# TODO: This is where most of the work will be done. You can change the model architecture,
#       add more layers, change activation functions, etc.
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.LeakyReLU()
        self.layer3= nn.Linear(128.64)
        self.bn3= nn.BatchNorm1d(64)
        self.relu3=nn.LeakyReLU(0.1)
        self.final_layer = nn.Linear(64, 1)
        self.output_layer=nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x= self.layer3(x)
        x= self.bn3(x)
        x= self.relu3(x)
        x= self.final_layer(x)
        x = self.output_layer(x)
        return x


def create_model(features):
    model = MyModel(features.shape[1])

    # define optimizer (feel free to change this)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer

if __name__ == '__main__':
    # create sample model with 228 input features
    model, _ = create_model(torch.zeros(1, 228))
    print(model)