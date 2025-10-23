import torch
from torch.utils.data import Dataset
import glob
import cv2
import skimage

device = 'cuda' if torch.cuda.is_available else 'cpu'

class inpaint_dataset(Dataset):
    def __init__(self, train, transform=None):
        super(inpaint_dataset, self).__init__()
        imgs = sorted(glob.glob('training_data/*.png')) if train else sorted(glob.glob('test_data/*.png'))
        img_size = 256
        self.img_files = []
        self.noisy_files = []
        self.transform = transform
        mask = torch.load('data/mask.npy')
        for i in range(int(len(imgs))):
            img = cv2.imread(imgs[i])
            img = skimage.color.rgb2gray(img)
            img = torch.from_numpy(img).view(1,1,img_size,img_size).float().to(device)
            self.img_files.append(img)
            self.noisy_files.append(mask*img+0.03*torch.randn_like(img))
        self.clean = torch.concat(self.img_files,0)
        self.noisy = torch.concat(self.noisy_files,0)

    def __getitem__(self, index):
        if self.transform is not None:
            clean = self.transform(self.clean[index])
            noisy = self.transform(self.noisy[index])
        else:
            clean = self.clean[index]
            noisy = self.noisy[index]

        return {'clean': clean,'noisy': noisy,'index': index}
    
    def __len__(self):
        return len(self.img_files)