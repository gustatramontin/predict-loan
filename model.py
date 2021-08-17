
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np

class NT(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NT, self).__init__()

        self.l1 = nn.Linear(input_size, 50)
        self.l2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

data = pd.read_csv('clientes.csv')

data = pd.Categorical(data['sexo'], categories=['Male', 'Female'])
data = data['estado_civil', 'working'].replace({
    "Yes": 1,
    "No": 0
})

data = pd.Categorical(data['educacao'], categories=['Graduate', 'Not Graduate'])
data = data['working'].replace({
    "Yes": 1,
    "No": 0
})

data = data['aprovacao_emprestimo'].replace({
    'Y': 1,
    'N': 0
})

data = pd.get_dummies(data)

train_dataset = np.array(data.drop(['aprovacao_emprestimo'], axis=0))

test_dataset = np.array(data['aprovacao_emprestimo'])





model = NT(input_size=input_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optmizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for id, (data, targets) in enumerate(train_loader):
        data = data.reshape(data.shape[0], -1, )

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optmizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optmizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    
    model.train()

    return float(num_correct) / float(num_samples) * 100

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
