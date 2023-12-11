import torch
import torch.nn as nn
import numpy as np

text = ["tranquil twilight of a picturesque autumn evening, the golden hues of the setting sun painted the sky with a breathtaking palette of warm oranges, fiery reds, and soft purples, casting a mesmerizing glow upon the gently ","a myriad of flora and fauna harmoniously coexisted, their existence weaving a tapestry of life that unfolded in the vast and untouched wilderness, where the rustling leaves whispered"]

# Build the input and target sequences
words = set(" ".join(text).split()) 
int2word = {i:w for i, w in enumerate(words)}
word2int = {w:i for i,w in enumerate(words)}
int2word[len(int2word)] = " "
word2int[" "] = len(word2int)

text = [sent.split() for sent in text]
input_seq = []
target_seq = []

for i in range(len(text)):
    input_seq.append(text[i][:-1])
    target_seq.append(text[i][1:])

#print(input_seq[0], target_seq[0])

maxlen = max([len(sent) for sent in text])
for i in range(len(text)):
    if len(input_seq[i])<maxlen:
        input_seq[i] += [" "]*(maxlen-len(input_seq[i])-1)
    if len(target_seq[i])<maxlen:
        target_seq[i] += [" "]*(maxlen-len(target_seq[i])-1)

for i in range(len(text)):
    input_seq[i] = [word2int[word] for word in input_seq[i]]
    target_seq[i] = [word2int[word] for word in target_seq[i]]

# Build the features

batch_size = len(text)
seq_len = maxlen-1 
dict_size = len(word2int)

def one_hot_encoding(sequence, dict_size, seq_len, batch_size):
    features = np.zeros((batch_size, seq_len, dict_size),dtype=np.float32)
    for i in range(batch_size):
        for u in range(seq_len):
            features[i,u,sequence[i][u]] = 1

    return features

input_features = one_hot_encoding(input_seq, dict_size, seq_len, batch_size)
input_features = torch.from_numpy(input_features)
target_features = torch.Tensor(target_seq)

#print(input_features.shape, target_features.shape)

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Build the RNN model
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
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

model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
model.to(device)
n_epochs = 100
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    input_features.to(device)
    output, hidden = model(input_features)
    loss = criterion(output, target_features.view(-1,).long())
    loss.backward()
    optimizer.step()
    
    if epoch%10==0:
        print("Epoch:{}/{}.....".format(epoch,n_epochs))
        print("Loss:{:.4f}".format(loss.item()))

# Predict output
def predict(sample, words):
    words = np.array([[word2int[w] for w in words]])
    words = one_hot_encoding(words, dict_size, words.shape[1], 1)
    words = torch.from_numpy(words)
    output, hidden = model(words)
    prob = nn.functional.softmax(output[-1],dim=0).data
    wordid = torch.max(prob, dim=0)[1].item()
    return int2word[wordid], hidden

def sample(model, out_len, start="a small"):
    model.eval()
    start = start.lower()
    words = [w for w in start.split()]
    size = out_len-len(words)
    for ii in range(size):
        word, h = predict(model, words)
        words.append(word)

    return " ".join(words)

print(sample(model,15,"the"))
