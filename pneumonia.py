# cnn with pneumonia data    97.2% accuracy (k=7)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir="./Data/Pneumonia/train"
test_dir="./Data/Pneumonia/test"

torch.manual_seed(42)

transform_train=transforms.Compose([ 
    transforms.Resize((150, 150)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomEqualize(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_test=transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomEqualize(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset=datasets.ImageFolder(train_dir,transform=transform_train)
test_dataset=datasets.ImageFolder(test_dir,transform=transform_test)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

print("Data Loaded")


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN,self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Dropout2d(0.3),
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.fc_layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,1)
        )
    
    def forward(self, x):
        x=self.conv_layers(x)
        x=self.fc_layers(x)
        return x


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=PneumoniaCNN().to(device)

print("CNN created")
print("PyTorch CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("Selected device:", torch.cuda.get_device_name(0))


criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1341/3875],device=device))  
optimizer=optim.Adam(model.parameters(),lr=0.0005,weight_decay=0.0001)

def train_model(model,train_loader,test_loader,criterion,optimizer,epochs):
    train_loss,test_loss,test_acc=[],[],[]
    for epoch in range(epochs):
        model.train()
        epoch_loss=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.float().to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs.squeeze(1),labels)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()*images.size(0)
        train_loss.append(epoch_loss/len(train_loader.dataset))
        
        model.eval()
        correct,total,test_epoch_loss=0,0,0
        with torch.no_grad():
            for images,labels in test_loader:
                images,labels=images.to(device),labels.float().to(device)
                outputs=model(images)
                test_loss_batch=criterion(outputs.squeeze(1),labels)
                test_epoch_loss+=test_loss_batch.item()*images.size(0)
                preds=(outputs.squeeze(1).sigmoid()>0.4).float()
                correct+=(preds==labels).sum().item()
                total+=labels.size(0)
        test_loss.append(test_epoch_loss/total)
        test_acc.append(correct/total)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, "
              f"Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]*100:.2f}%")
    
    return train_loss,test_loss,test_acc

train_loss,test_loss,test_acc=train_model(model,train_loader,test_loader,criterion,optimizer,epochs=35)
