import numpy as np
import torch 
import torch.nn as nn

# Make the training batches
def make_batch():
    target_batch, input_batch = [], []
    
    words = sentence.split()
    # Add an iterator if multiple sentence is present
    input = [word_dict[w] for w in words[:-1]]
    target = word_dict[words[-1]]
    input_batch.append(np.eye(n_class)[input])
    target_batch.append(target)

    return input_batch, target_batch

# Make the model
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True) 
        self.W = nn.Linear(n_hidden*2, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self,X):
        input = X.transpose(0,1) # [n_step, batch_size, n_class]
        hidden_state = torch.zeros(1*2, len(X), n_hidden) # [n_layers*2, batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), n_hidden)

        outputs, (_,_) = self.lstm(input,(hidden_state, cell_state))
        outputs = outputs[-1]
        model = self.W(outputs) + self.b
        return model

# Main function
if __name__=="__main__":
    n_hidden = 5
    sentence = ('Lorem ipsum dolor sit amet consectetur adipisicing elit '
        			'sed do eiusmod tempor incididunt ut labore et dolore magna '
                'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    			)
    word_dict = {w:i for i,w in enumerate(list(set(sentence.split())))}
    number_dict = {i:w for i,w in enumerate(list(set(sentence.split())))}
    
    n_class = len(word_dict)
    max_len = len(sentence.split())
    lr = 0.01
    model = BiLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_batch, target_batch = make_batch() 
    input_batch = torch.FloatTensor(input_batch) # torch.tensor(np.asfarray(input_batch))
    target_batch = torch.LongTensor(target_batch) # torch.tensor(np.asfarray(target_batch))
    
    for epoch in range(1,11):
        optimizer.zero_grad()
        output = model(input_batch)
        loss =  criterion(output, target_batch)
        print("Epoch: {}, cost: {:.4f}".format(epoch, loss.item()))
        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1,keepdim=True)[1]
    print(sentence)
    print(number_dict[predict.squeeze().item()])
    #print([number_dict[n.item()] for n in predict.squeeze()])
