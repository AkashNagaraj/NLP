import torch
from torch import nn
import numpy as np

# Read the data and form the mappinng to integers
text = ["webpage to generate any number of random sentence", "tool can be used for a variety of purposes, including"]
chars = set(" ".join(text))
int2chars = {i:w for i,w in enumerate(chars)}
char2int = {c:i for i,c in enumerate(chars)}

maxlen = len(max(text,key=len))
for i in range(len(text)):
    while len(text[i])<maxlen:
        text[i] += " "

# Build the sequence that is to be passed
input_seq = []
target_seq = []

for i in range(len(text)):
    input_seq.append(text[i][:-1])
    target_seq.append(text[i][1:])

for i in range(len(text)):
    input_seq[i] = [char2int[char] for char in input_seq[i]]
    target_seq[i] = [char2int[char] for char in target_seq[i]]

# Create the input features

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encoding(sequence, dict_size, seq_len, batch_size):
    features = np.zeros((batch_size, seq_len, dict_size),dtype = np.float32)
    
    for i in range(batch_size):
        for u in range(seq_len):
            features[i,u,sequence[i][u]] = 1
    return features

input_seq = one_hot_encoding(input_seq, dict_size, seq_len, batch_size)
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

print("Input shape : {}, Target shape : {}".format(input_seq.shape, target_seq.shape))

# Use cuda
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# Build the model

class Model(nn.Module):
    def __init__(self,input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, X):
        batch_size = X.shape[0]
        hidden = self.init_hidden(batch_size)
        output, hidden = self.rnn(X, hidden)
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

# Train the model
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim = 12, n_layers = 1)
model.to(device)
n_epochs = 10
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    input_seq.to(device)
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward()
    optimizer.step()
    
    print("Epoch : {}/{}.....".format(epoch,n_epochs))
    print("Loss : {:.4f}".format(loss.item()))
