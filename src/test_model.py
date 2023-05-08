import torch
import torch.optim as optim
import os
from tqdm.auto import tqdm
from model import GNNImageClassificator
from datasets import build_train_val_dataloaders

best_model_saved_path = 'model_weight'

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# build the model
model = GNNImageClassificator(in_channels=3, hidden_dim=64).to(device)
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

    prediction_test = {}

    model.eval()
    print('Testing')
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            label_list = []
            classes_dict = {}
            pred_dict = {}
            counter += 1

            data_node_features = data["batch_node_features"]
            data_edge_indices = data["batch_edge_indices"]
            classes = data["classes"]

            # forward pass
            outputs = model(batch_node_features=data_node_features, batch_edge_indices=data_edge_indices)
            
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            
            classes_dict['true_label'] = classes
            pred_dict['predict_label'] = preds
            label_list.append(classes_dict)
            label_list.append(pred_dict)

            valid_running_correct += (preds == classes).sum().item()
            prediction_test[f'epoch_{i}'] = label_list

    # loss and accuracy for the complete epoch
    final_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return final_acc, prediction_test

# test the last epoch saved model
def test_last_model(model, checkpoint, test_loader):
    print('Loading last epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    lst_acc, prediction_result_last_model = test(model, test_loader)
    print(f"Last epoch saved model accuracy: {lst_acc:.3f}")
    return lst_acc, prediction_result_last_model

# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    bst_acc, prediction_result_best_model = test(model, test_loader)
    print(f"Best epoch saved model accuracy: {bst_acc:.3f}")
    return bst_acc, prediction_result_best_model

if __name__ == '__main__':
    lst_acc, prediction_result_last_model = test_last_model(model, last_model_cp, test_loader)
    bst_acc, prediction_result_best_model = test_best_model(model, best_model_cp, test_loader)