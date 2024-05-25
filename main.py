import dataset
from model import CharRNN, CharLSTM

# import some packages you need here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from dataset import Shakespeare
from model import CharRNN, CharLSTM
from generate import generate
import warnings 
warnings.filterwarnings(action='ignore')

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    # write your codes here
    model.train()
    total_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs, model.init_hidden(inputs.size(0)))
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    trn_loss = total_loss / len(trn_loader)

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs, model.init_hidden(inputs.size(0)))
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
            
    val_loss = total_loss / len(val_loader)
    
    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    input_file = 'C:\\Users\\TEEM\\Desktop\\shakespears\\shakespeare_train.txt'
    batch_size = 128
    sequence_length = 30
    num_epochs = 10
    learning_rate = 0.0001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = Shakespeare(input_file)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    
    criterion = nn.CrossEntropyLoss()

    # RNN
    rnn_model = CharRNN(input_size=len(dataset.char2idx), hidden_size=128, output_size=len(dataset.char2idx), n_layers=2).to(device)
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    rnn_train_losses, rnn_val_losses = [], []

    for epoch in range(num_epochs):
        rnn_train_loss = train(rnn_model, trn_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        rnn_train_losses.append(rnn_train_loss)
        rnn_val_losses.append(rnn_val_loss)
 
    
    # LSTM
    lstm_model = CharLSTM(input_size=len(dataset.char2idx), hidden_size=128, output_size=len(dataset.char2idx), n_layers=2).to(device)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    lstm_train_losses, lstm_val_losses = [], []

    for epoch in range(num_epochs):
        lstm_train_loss = train(lstm_model, trn_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)
        lstm_train_losses.append(lstm_train_loss)
        lstm_val_losses.append(lstm_val_loss)
        
        
    # Plotting
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, rnn_train_losses, label='RNN Train Loss')
    plt.plot(epochs, rnn_val_losses, label='RNN Val Loss')
    plt.plot(epochs, lstm_train_losses, label='LSTM Train Loss')
    plt.plot(epochs, lstm_val_losses, label='LSTM Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    
    
    # Generate samples
    seed_characters = "The characters"
    temperature = 0.7
    rnn_samples = generate(rnn_model, seed_characters, temperature, device, dataset.char2idx, dataset.idx2char)
    lstm_samples = generate(lstm_model, seed_characters, temperature, device, dataset.char2idx, dataset.idx2char)
    
    print("\nGenerated Samples with RNN:\n")
    for sample in rnn_samples:
        print(sample)
    
    print("\nGenerated Samples with LSTM:\n")
    for sample in lstm_samples:
        print(sample)    

if __name__ == '__main__':
    main()