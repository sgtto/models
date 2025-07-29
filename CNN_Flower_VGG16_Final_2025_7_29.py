
# coding: utf-8

# In[ ]:


import torch
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import numpy as np
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

torch.manual_seed(114514)
np.random.seed(114514)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.vgg16(pretrained=True)
model.classifier[6]=nn.Linear(4096,5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
val_preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder('/kaggle/input/flowers-datasets/PR_hw/data2/flowers_train', transform=train_preprocess)
val_dataset = datasets.ImageFolder('/kaggle/input/flowers-datasets/PR_hw/data2/flowers_test', transform=val_preprocess)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names=train_dataset.classes
num_class=len(class_names)
all_labels = []
all_preds = []
def training(model,train_loader,criterion,optimizer,scheduler,epochs):                                           
    model.train()
    best_acc=0
    best_para=None
    for epoch in range(epochs):
        total_loss=0
        correct=0
        acc=0
        for train_image,labels in train_loader: 
            train_image=train_image.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model(train_image)
            _,preds=torch.max(outputs,1)
            loss=criterion(outputs,labels)
            correct+=torch.sum(preds==labels.data)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc=correct.double().cpu().item()/len(train_dataset)
        if best_acc<acc:
            best_acc=acc
            best_para=copy.deepcopy(model.state_dict())
        print(f"Epoch [{epoch+1}/{epochs}], best_accuracy: {best_acc:.4f}")
    return best_para,best_acc
    
def evaluate(model,val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_image,labels in val_loader:
            val_image=val_image.to(device)
            labels=labels.to(device)
            output=model(val_image)
            _,predicted=torch.max(output,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    print(f'Accuracy: {100 * correct / total}%')
    return correct / total
print('start!')
def simple_tsne(model, dataloader, class_names, device, perplexity=25):

    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)

            feat = model.features(imgs)
            feat = torch.flatten(feat, 1)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)
    

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_feat = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        mask = (labels == i)
        plt.scatter(tsne_feat[mask, 0], tsne_feat[mask, 1], 
                    label=name, alpha=0.7, s=30)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.title("t-SNE Visualization")
    plt.tight_layout()  # 自动调整布局
    plt.show()
model=model.to(device)
best_para, best_acc =training(model,train_loader,criterion,optimizer,exp_lr_scheduler,20)
print(best_acc)
print('evaluating')
model.load_state_dict(best_para)
evaluate(model,val_loader)
torch.save(model.state_dict(),'flower_vgg16.pth')
cm=confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(10,8))
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title('confusion matrix')
plt.tight_layout()
plt.savefig('flower confusion matrix.png')
plt.show()
print("that is all")
simple_tsne(
    model=model,      
    dataloader=val_loader,
    class_names=class_names, 
    device=device
)

