

import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#hyperparameters
epochs = 100
test_size = 0.2


df = pd.read_csv('dataset_parameters.csv')
df = df.sample(frac=1)


X = df.iloc[:, :-1]
y = df.iloc[:, -1].values
X['MAR'] = X['MAR'].apply(lambda x : 1 if x >= 0.35 else 0)
X = X.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# Compute the mean and standard deviation of the training data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Normalize the training and test data using the mean and standard deviation
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train_norm, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test_norm, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


Para = pd.DataFrame().assign(mean=mean, std=std)
Para.to_csv('Parameters.csv', header=False, index=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 32)
        self.fc10 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.sigmoid(self.fc10(x))
        return x


net = Net()


criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())


for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(X_train)
    loss = criterion(output, y_train.unsqueeze(1))
    #print( loss)
    print(epoch, ' ', loss)
    loss.backward()
    optimizer.step()


torch.save(net.state_dict(), 'model_weights.pth')


net.eval()  # set the model to evaluation mode
with torch.no_grad():
    output = net(X_test)
    predictions = output.round()  # round the output to 0 or 1
    accuracy = (predictions == y_test.unsqueeze(1)).float().mean()  # compute accuracy
print("Accuracy: {:.2f}%".format(accuracy.item() * 100))
print(confusion_matrix(y_test, predictions))