import dataset as ds
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_pth = 'weight/resnet50_0.9616.pkl'
BS = 128

model = torch.load(weight_pth).to(device)
test_dataset = ds.self_Cifar(device, mode='test')
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BS,
                        shuffle=False)
with torch.no_grad():
    y_hat = None
    for x, y in tqdm(test_loader):
        x = x.half()
        y = y.view(-1)
        output = model(x)
        _, pred = torch.max(output.detach(), 1)
        if y_hat is None:
            y_hat = pred
        else:
            y_hat = torch.cat((y_hat, pred), 0)

    y_pred = y_hat.cpu().numpy()
    y_test = np.load("y_test.npy")
    print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))