import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np

# Read the data
filename = "data/book.txt"
raw_text = open(filename, 'r', encoding="utf-8").read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char2int = {c:i for i,c in enumerate(chars)}
int2char = {i:c for i,c in enumerate(chars)}

n_chars = len(raw_text)
n_vocab = len(char2int)
print("Total character: ", n_chars)
print("Total Vocab: ",n_vocab)

# Build the data sequence
seq_length = 100
use_seq = 100
raw_text = raw_text[:len(raw_text)//seq_length*seq_length]
raw_text = raw_text[:seq_length*use_seq]
n_chars = len(raw_text)
print("The characters that are used for training",n_chars)

dataX = []
datay = []
for i in range(0, n_chars-seq_length, 1):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([char2int[s] for s in seq_in])
    datay.append(char2int[seq_out])
n_patterns = len(dataX)

X = torch.tensor(dataX,dtype=torch.float32).reshape(n_patterns, seq_length, 1) # Is 1 the input size?
X = X/float(n_vocab)
y = torch.tensor(datay)
print("Size of input tensor : {}, size of output tensor : {}".format(X.shape, y.shape))

# Build Model
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=20, num_layers=1, batch_first=True) # [input features, h_cells per  layer, n_layers]
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(20, n_vocab)
    
    def forward(self,X):
        x, _ = self.lstm(X) # contains output from each cell - x
        x = x[:, -1, :] # ONLY LAST OUTPUT
        x = self.linear(self.dropout(x))
        return x

# Training
n_epochs = 1
batch_size = 50
lr = 0.01
model = CharModel()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss(reduction="sum") # What does this mean?
loader = data.DataLoader(data.TensorDataset(X,y), shuffle=True, batch_size=batch_size)
best_model = None
best_loss = np.inf

for epoch in range(1,n_epochs+1):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    #if epoch%10==0:
    print("Epoch : {}/{}....".format(epoch, n_epochs))
    print("Loss is :",loss.item())

# Predict and Generate
start = np.random.randint(0,len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char2int[c] for c in prompt]
print("The input is : ", prompt)
#print("The pattern is: ",pattern)

gen_length = 200
model.eval()
with torch.no_grad():
    for i in range(gen_length):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab) # (batch_size, seq_length, output_size)
        x = torch.tensor(x,dtype=torch.float32)
        prediction = model(x)
        index = int(prediction.argmax())
        result = int2char[index]
        print(result, end="")
        pattern.append(result)
        pattern = pattern[1:]
print()
print("Done")
