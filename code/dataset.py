import os
import csv
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

#Trasformazione sulle immagini
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class LFWDataLoader(Dataset):
    def __init__(self, root_dir, match_file, mismatch_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []

         # Carica le coppie positive (matchpairs)
        with open(match_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                person = row['name']
                img1 = row['imagenum1']
                img2 = row['imagenum2']
                path1 = os.path.join(root_dir, person, f"{person}_{str(img1).zfill(4)}.jpg")
                path2 = os.path.join(root_dir, person, f"{person}_{str(img2).zfill(4)}.jpg")
                label = 1
                self.pairs.append((path1, path2, label))
        # Carica le coppie negative (se hai un mismatch.csv simile)
        with open(mismatch_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                person1 = row['name1']
                img1 = row['imagenum1']
                person2 = row['name2']
                img2 = row['imagenum2']
                path1 = os.path.join(root_dir, person1, f"{person1}_{str(img1).zfill(4)}.jpg")
                path2 = os.path.join(root_dir, person2, f"{person2}_{str(img2).zfill(4)}.jpg")
                label = 0  # Coppia negativa
                self.pairs.append((path1, path2, label))
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        # Applica la trasformazione per convertire in tensori
        img1 = transform(img1)
        img2 = transform(img2)

        # Controllo aggiuntivo per la conversione
        assert isinstance(img1, torch.Tensor), f"L'immagine 1 non è un tensore: {path1}"
        assert isinstance(img2, torch.Tensor), f"L'immagine 2 non è un tensore: {path2}"

        return img1, img2, torch.tensor(label, dtype=torch.float32)

