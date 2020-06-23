from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from torchvision import transforms

class self_Cifar(Dataset):
    def __init__(self, device, mode='train'):
        super().__init__()
        self.x, self.y = np.load(f'./x_{mode}.npy'), np.load(f'./y_{mode}.npy')
        self.y = torch.from_numpy(self.y)
        self.device = device

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((-40, 40)),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        x = self.transform(x)
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.y)