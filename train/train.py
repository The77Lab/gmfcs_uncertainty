
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DataSet_Classification, DataSet_Regression
import math
import os
import sys
# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

# from models.heads import Classifier
from models.stgcn import STGCN


def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class STGCN_Regressor(nn.Module):

    def __init__(self,
                 backbone,
                 dropout=0.5,
                 init_std=0.01):
        super(STGCN_Regressor, self).__init__()

        args = backbone.copy()
        args.pop('type')
        self.backbone = STGCN(**args)
        # self.cls_head = Classifier(
        #     num_classes=num_classes, dropout=0.5, latent_dim=512)

        self.latent_dim = 512
        self.in_channels = 256
        self.dropout = dropout
        self.init_std = init_std

        self.pool = nn.AvgPool2d((31, 17))
        
        self.linear1 = nn.Linear(self.in_channels, self.latent_dim)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.out_mean = nn.Linear(self.latent_dim, 1)

        self.linear2 = nn.Linear(self.in_channels, self.latent_dim)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.out_log_var = nn.Linear(self.latent_dim, 1)

        normal_init(self.linear1, std=self.init_std)
        normal_init(self.out_mean, std=self.init_std)
        normal_init(self.linear2, std=self.init_std)
        normal_init(self.out_log_var, std=self.init_std)


    def forward(self, keypoint):
        """Define the computation performed at every call."""
        x = self.backbone(keypoint)
        # print(x.shape)
        N, M, C, T, V = x.shape
        x = x.reshape(N, C, T, V)
        x = self.pool(x)
        x = x.reshape(N, C)

        m = self.linear1(x)
        m = self.dropout1(m)
        m = F.relu(m)
        m = self.out_mean(m)

        log_var = self.linear2(x)
        log_var = self.dropout2(log_var)
        log_var = F.relu(log_var)
        log_var = self.out_log_var(log_var)

        return m, log_var


batch_size = 128
sample_folder = 'path to the dataset'


backbone_cfg = {
    'type': 'STGCN',
    'gcn_adaptive': 'init',
    'gcn_with_res': True,
    'tcn_type': 'mstcn',
    'graph_cfg': {
        'layout': 'coco',
        'mode': 'spatial'
    },
    'pretrained': None
}

device = 'cuda:0'


            
# Load pre-trained weights to the backbone
backbone_state_dict = os.path.join(script_dir, 'j.pth')
# load_checkpoint(model.backbone, backbone_state_dict)
tmp = torch.load(backbone_state_dict)
# print(tmp.keys())

del tmp['cls_head.fc_cls.weight']
del tmp['cls_head.fc_cls.bias']

def my_relaxed_mse(out_m, target, margin):
    dist = torch.abs(out_m-target)
    # print(dist - margin)
    # set the dist margin, if dist < margin, set dist to 0, else dist = dist - margin, use F.relu
    dist = F.relu(dist - margin)
    # calculate the mse loss
    return torch.mean(dist**2)


def my_relaxed_l1(out_m, target, margin):
    dist = torch.abs(out_m-target)
    # print(dist - margin)
    # set the dist margin, if dist < margin, set dist to 0, else dist = dist - margin, use F.relu
    dist = F.relu(dist - margin)
    # calculate the mse loss
    return torch.mean(dist)

def my_relaxed_mse_full(out_m, out_log_var, target, margin):
    # return torch.mean(0.5*torch.exp(-out_log_var)*(out_m-target)**2 + 0.5*out_log_var)
    return torch.mean(0.5*torch.exp(-out_log_var)*(F.relu(torch.abs(out_m-target) - margin))**2 + 0.5*out_log_var)

def my_relaxed_l1_full(out_m, out_log_var, target, margin):
    # return torch.mean(0.5*torch.exp(-out_log_var)*(out_m-target)**2 + 0.5*out_log_var)
    return torch.mean(0.5*torch.exp(-out_log_var)*(F.relu(torch.abs(out_m-target) - margin)) + 0.5*out_log_var)

margin = 0.1
losstype = 'mse'
augmented= True
ratio = 1

print('margin: ', margin, ' losstype: ', losstype, ' augmented: ', augmented, ' ratio: ', ratio)
model_save_path = os.path.join(script_dir, 'model', 'model_margin_' + str(margin) + '_losstype_' + losstype + '_augmented_' + str(augmented) + '_ratio_' + str(ratio) + '.pth')

if augmented:
    train_dataset = DataSet_Regression(os.path.join(script_dir, 'train_dataset14.npy'),
                            sample_folder, data_augmentation=True, replicated=True)
else:
    train_dataset = DataSet_Regression(os.path.join(script_dir, 'train_dataset14.npy'),
                            sample_folder, data_augmentation=False, replicated=False)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = DataSet_Regression(os.path.join(script_dir, 'val_dataset14.npy'),
                    sample_folder, data_augmentation=False, replicated=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = STGCN_Regressor(backbone=backbone_cfg)

model.load_state_dict(tmp, strict=False)

for param in model.backbone.parameters():
    param.requires_grad = False

for param in model.linear2.parameters():
    param.requires_grad = False

for param in model.out_log_var.parameters():
    param.requires_grad = False

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

# val_best_acc = -math.inf
val_best_loss = math.inf

# criterion_1 = nn.MSELoss()
if losstype == 'mse':
    criterion_1 = my_relaxed_mse
    criterion_2 = my_relaxed_mse_full

unfreeze_backbone_epoch = 2
uncertainty_epoch = 4

num_epochs = 30
for epoch in range(num_epochs):
    if epoch == unfreeze_backbone_epoch:
        for layer in range(9, 6, -1):
            for param in model.backbone.gcn[layer].parameters():
                param.requires_grad = True
    if epoch == uncertainty_epoch:
        for param in model.linear2.parameters():
            param.requires_grad = True
        for param in model.out_log_var.parameters():
            param.requires_grad = True

    
    train_loss = 0.0
    train_correct = 0
    model.train()  # set the model to train mode
    epoch_pbar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs}",
                    total=len(train_dataloader.dataset) / batch_size, position=0)

    # epoch_acc = 0.0
    epoch_loss = 0.0

    sample_count = 0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.type(torch.FloatTensor)
        labels = labels.unsqueeze(1)
        labels = labels.to(device)
        optimizer.zero_grad()
        # print(inputs.shape)
        inputs = inputs.view(-1, 1, 124, 17, 3)
        # print(labels.shape)
        if augmented:
            # replicate labels
            labels = labels.repeat(2, 1)
            # print(labels.shape)
        # print(inputs.shape)
        # Forward pass
        # print(inputs.shape, labels.shape)
        outputs_m, outputs_log_var = model(inputs)
        # print(outputs_m.shape, outputs_log_var.shape, labels.shape)
        # print(outputs)
        if epoch < uncertainty_epoch:
            loss = criterion_1(outputs_m, labels, margin)
        else:
            loss = criterion_2(outputs_m, outputs_log_var, labels, margin)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        sample_count += batch_size

        epoch_loss += loss.item()

        # epoch_acc = train_correct * 100.0 / sample_count
        tmp_loss = epoch_loss * 1.0 / sample_count * batch_size

        # Update the progress bar for the epoch
        epoch_pbar.update(1)
        # epoch_pbar.set_postfix({'loss': tmp_loss, 'acc': epoch_acc})
        epoch_pbar.set_postfix({'loss': tmp_loss})

    # Compute the training loss and accuracy for this epoch
    epoch_loss /= (len(train_dataloader.dataset) / batch_size)
    # train_accuracy = 100.0 * train_correct / len(train_dataloader.dataset)
    # Close the progress bar for the epoch
    epoch_pbar.close()
    # print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
    print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}')


    # Evaluate the model on the validation set
    val_loss = 0.0
    # val_correct = 0
    model.eval()  # set the model to eval mode

    epoch_pbar = tqdm(desc=f"VAL Epoch {epoch+1}/{num_epochs}",
                    total=len(val_dataloader.dataset) / batch_size, position=0)
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.FloatTensor)
            labels = labels.unsqueeze(1)
            labels = labels.to(device)

            # Forward pass
            outputs_m, outputs_log_var = model(inputs)
            # loss = criterion_1(outputs_m, labels, margin)
            loss = criterion_2(outputs_m, outputs_log_var, labels, margin)

            val_loss += loss.item()

            # Compute the number of correctly classified samples
            # _, predicted = torch.max(outputs.data, 1)
            # val_correct += (predicted == labels).sum().item()
            epoch_pbar.update(1)
    epoch_pbar.close()

    # Compute the validation loss and accuracy for this epoch
    val_loss /= (len(val_dataloader.dataset) / batch_size)
    # val_accuracy = 100.0 * val_correct / len(val_dataloader.dataset)
    # print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}')


    if val_loss < val_best_loss and epoch >= uncertainty_epoch:
        val_best_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print('model saved!')

    # update the learning rate
    scheduler.step()
