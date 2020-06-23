import dataset as ds
import models
import torchvision.models as tmodels
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm

Epoch = 200
BS = 128
LR = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'resnet50'
file_name = f'{model_name}_lr_{LR}_BS_{BS}'

# model = models.Model().to(device).half()
model = models.cifar10_resnet_50(pretrained=True).to(device).half()
opt = torch.optim.SGD([
    {'params': model.resnet_50.parameters(), 'lr': 0.005},
    {'params': model.classify.parameters(), 'lr': LR}
], weight_decay=0.01)

train_dataset = ds.self_Cifar(device, mode='train')
test_dataset = ds.self_Cifar(device, mode='test')
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BS,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BS,
                        shuffle=False)

Loss = nn.CrossEntropyLoss()

print(f'{model_name}')
print(f'Length of train data: {len(train_dataset)}')
print(f'Length of test data: {len(test_dataset)}')

best_test_acc = 0
for epoch in range(Epoch):
    print(f'Epoch {epoch}')
    print(f'training')
    model.train()
    train_tot_loss, train_cnt = 0, 0
    correct = 0
    for x, y in tqdm(train_loader):
        train_cnt += 1
        x = x.half()
        y = y.view(-1)
        output = model(x)
        
        loss = Loss(output, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_tot_loss += loss.detach().cpu().item()
        _, pred = torch.max(output.detach(), 1)
        correct += (pred==y.detach()).sum().item()

    del x
    del y

    train_avg_loss = train_tot_loss/train_cnt
    train_accuracy = correct/len(train_dataset)
    print(f'{correct}')
    print(f'train accuracy: {train_accuracy}, train loss: {train_avg_loss}')

    print(f'testing')
    model.eval()
    test_tot_loss, test_cnt = 0, 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.half()
            test_cnt += 1
            y = y.view(-1)
            output = model(x)
            loss = Loss(output, y)
            test_tot_loss += loss.detach().cpu().item()
            _, pred = torch.max(output.detach(), 1)
            correct += (pred==y.detach()).sum().item()
        del x
        del y

    test_avg_loss = test_tot_loss/test_cnt
    test_accuracy = correct/len(test_dataset)
    print(f'test accuracy: {test_accuracy}, test loss: {test_avg_loss}')

    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        torch.save(model, f'{model_name}_{best_test_acc}.pkl')

    with open(f'{file_name}_train_acc', 'a') as f:
        f.write(f'{epoch}: {train_accuracy}\n')
    with open(f'{file_name}_train_loss', 'a') as f:
        f.write(f'{epoch}: {train_avg_loss}\n')
    with open(f'{file_name}_test_acc', 'a') as f:
        f.write(f'{epoch}: {test_accuracy}\n')
    with open(f'{file_name}_test_loss', 'a') as f:
        f.write(f'{epoch}: {test_avg_loss}\n')