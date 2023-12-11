import torch
import torch.nn as nn
import numpy as np

# Build the sequence
def make_batch(seq_data):
    input_batch, target_batch = [], []
    for i in range(len(seq_data)):
        input_batch.append([word2int[word] for word in seq_data[i][:-1]])
        target_batch.append([word2int[word] for word in seq_data[i][1:]])
    return input_batch, target_batch

# Build the features
def one_hot_encode(sequence, batch_size, seq_len, n_class):
    features = np.zeros((batch_size, seq_len, n_class),dtype=np.float32)
    for i in range(batch_size):
        for u in range(seq_len):
            features[i,u,sequence[i][u]] = 1
    return features
 
# Build the model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias = False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        input = X.transpose(0,1) # X : [n_step, batch_Size, n_class]
        hidden_state = torch.ones(1,len(X),n_hidden) # [n_layers, batch_Size, n_hidden]
        cell_state = torch.zeros(1, len(X), n_hidden) 
        outputs, (_,_) = self.lstm(input, (hidden_state, cell_state)) 
        outputs = outputs[-1]
        model = self.W(outputs) + self.b # [batch_size, n_class]
        return model

# Train the model

# Generate text from the trained model

if __name__=="__main__":
    n_steps = 3 # number of cells
    n_hidden = 12 # number of hidden units in a cell
    
    text = ["make", "need", "coal", "love"]
    maxlen = max([len(w) for w in text])
    char_arr = [c for c in "abcdefghijklmnopqrstuvwxyz"]
    word2int = {c:i for i,c in enumerate(char_arr)}
    int2word = {i:c for i,c in enumerate(char_arr)}
    n_class = len(word2int)

    input_batch, target_batch = make_batch(text)
    input_batch = one_hot_encode(input_batch, batch_size=len(text), seq_len=maxlen, n_class=len(word2int))
    input_batch = torch.from_numpy(input_batch)
    target_batch = torch.Tensor(target_batch)
    
    lr = 0.001
    model = LSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1,11):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)

        print("Loss : {:4f}".format(loss.item()))
        loss.backward()
        optimizer.step()

