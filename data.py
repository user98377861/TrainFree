from torch.utils.data import Dataset
from PIL import Image

class ArtifactDataset(Dataset):
    def __init__(self, dataframe, transform=None, img_key='image_path'):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, img_key='image_path'):
        image_path = self.df.iloc[idx][img_key]
        label = self.df.iloc[idx]['label']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class BinaryDataset(Dataset):
    def __init__(self, dataframe1, dataframe2, transform=None, img_key='image_path'):
        self.df1 = dataframe1
        self.df2 = dataframe2

    def __len__(self):
        return len(self.df1)+len(self.df2)

    def __getitem__(self, idx, img_key='image_path'):
        if idx<len(self.df1):
            image_path = self.df1.iloc[idx][img_key]
        else:
            image_path = self.df2.iloc[idx-len(self.df1)][img_key]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
