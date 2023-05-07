import torch
import torch.optim as optim
import os
from tqdm.auto import tqdm
from model import GNNImageClassificator
from datasets import build_train_val_dataloaders

best_model_saved_path = 'src/best_model'

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# build the model
model = GNNImageClassificator(in_channels=3, hidden_dim=152).to(device)
# load the best model checkpoint
best_model_cp = torch.load(os.path.join(best_model_saved_path,'best_model.pth'))
best_model_epoch = best_model_cp['epoch']
print(f"Best model was saved at {best_model_epoch} epochs\n")
# load the last model checkpoint
last_model_cp = torch.load(os.path.join(best_model_saved_path,'final_model.pth'))
last_model_epoch = last_model_cp['epoch']
print(f"Last model was saved at {last_model_epoch} epochs\n")
# get the test dataset and the test data loader
train_loader, valid_loader, test_loader = build_train_val_dataloaders()

def test(model, testloader):
    """
    Function to test the model
    """
    # set model to evaluation mode
    model.eval()
    print('Testing')
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            data_node_features = data["batch_node_features"]
            data_edge_indices = data["batch_edge_indices"]
            classes = data["classes"]

            # forward pass
            outputs = model(batch_node_features=data_node_features, batch_edge_indices=data_edge_indices)

            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == classes).sum().item()
        
    # loss and accuracy for the complete epoch
    final_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return final_acc

# test the last epoch saved model
def test_last_model(model, checkpoint, test_loader):
    print('Loading last epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = test(model, test_loader)
    print(f"Last epoch saved model accuracy: {test_acc:.3f}")
# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = test(model, test_loader)
    print(f"Best epoch saved model accuracy: {test_acc:.3f}")

if __name__ == '__main__':
    test_last_model(model, last_model_cp, test_loader)
    test_best_model(model, best_model_cp, test_loader)