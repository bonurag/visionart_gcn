
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from model import GNNImageClassificator
from datasets import build_train_val_dataloaders
from utils import save_model, save_plots, SaveBestModel

from tensorboardX import SummaryWriter

# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        data_node_features = data["batch_node_features"]
        data_edge_indices = data["batch_edge_indices"]
        classes = data["classes"]

        optimizer.zero_grad()
        # forward pass
        outputs = model(batch_node_features=data_node_features, batch_edge_indices=data_edge_indices)
        # calculate the loss
        loss = criterion(outputs, classes)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == classes).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    train_writer.add_scalar('train_accuracy', epoch_acc, counter)
    train_writer.add_scalar('train_loss', epoch_loss, counter)
    train_writer.add_scalar('batch', counter, counter)

    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            data_node_features = data["batch_node_features"]
            data_edge_indices = data["batch_edge_indices"]
            classes = data["classes"]

            # forward pass
            logits = model(batch_node_features=data_node_features, batch_edge_indices=data_edge_indices)
            # calculate the loss
            loss = criterion(logits, classes)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(logits.data, 1)
            valid_running_correct += (preds == classes).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    val_writer.add_scalar('val_accuracy', epoch_loss, counter)
    val_writer.add_scalar('val_loss', epoch_loss, counter)
    val_writer.add_scalar('batch', counter, counter)

    return epoch_loss, epoch_acc

tensorboard_saved_path = 'src/tensorboard_data'
os.makedirs(tensorboard_saved_path, exist_ok=True)

train_writer = SummaryWriter(os.path.join(tensorboard_saved_path, 'train'), 'train')
val_writer = SummaryWriter(os.path.join(tensorboard_saved_path, 'val'), 'val')

# get the training and validaion data loaders
train_loader, valid_loader, test_loader = build_train_val_dataloaders()

# learning_parameters 
lr = 1e-3
epochs = 64

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# build the model
model = GNNImageClassificator(in_channels=3, hidden_dim=152).to(device)
print(model)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()
# initialize SaveBestModel class
save_best_model = SaveBestModel()

# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

    # save the best model till now if we have the least loss in the current epoch
    save_best_model(
        valid_epoch_loss, epoch, model, optimizer, criterion
    )
    print('-'*50)
    
# save the trained model weights for a final time
save_model(epochs, model, optimizer, criterion)
# save the loss and accuracy plots
save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('TRAINING COMPLETE')