import random
import torch
from torch.utils.data import DataLoader, Subset
from ResNet import ResNet, resnet18
from loss import ElasticFaceLoss, ContrastiveLoss
from dataset import LFWDataLoader
import torchvision.transforms as transforms
import torch.optim as optim 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#DataLoader
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),       # Flip orizzontale casuale
    transforms.RandomRotation(10),                # Rotazione casuale fino a ±10 gradi
    transforms.ColorJitter(brightness=0.2,        # Variazione della luminosità
                           contrast=0.2, 
                           saturation=0.2),
    transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),  # Ritaglio casuale
    transforms.ToTensor(),                        # Converti l'immagine in tensore
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizzazione
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#Training set 
train_data = LFWDataLoader(
    root_dir='../DataSet/archive/lfw-deepfunneled/lfw-deepfunneled',
    match_file='../DataSet/archive/matchpairsDevTrain.csv',
    mismatch_file='../DataSet/archive/mismatchpairsDevTrain.csv',
    transform=train_transform
)

#match_indices = list(range(1100))
#mismatch_indices = list(range(1100,2200))

#random.shuffle(match_indices)
#random.shuffle(mismatch_indices)

#subset_size = 1100
#subset_indices = match_indices[:subset_size // 2] + mismatch_indices[:subset_size // 2]

#balanced_subset = Subset(train_data, subset_indices)
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True, num_workers=3)

#Validation set
val_data = LFWDataLoader(
    root_dir='../DataSet/archive/lfw-deepfunneled/lfw-deepfunneled',
    match_file='../DataSet/archive/matchpairsDevTest.csv',
    mismatch_file='../DataSet/archive/mismatchpairsDevTest.csv',
    transform=val_transform
)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
model = resnet18(num_classes=128).to(device)

#Loss e opt
criterion = ElasticFaceLoss(s = 64.0, m = 0.35, std = 0.05)
#criterion = ContrastiveLoss()

optimizer = optim.Adam(model.parameters(),lr = 0.0005, weight_decay=0.0001)

#Training loop
for epoch in range(40):
    model.train()
    loss_corrente = 0.0

    for img1, img2, labels in train_loader:
        img1,img2, labels = img1.to(device),img2.to(device),labels.to(device)
        #print(f"Caricamento batch completato. Dimensioni batch: {img1.shape}, {img2.shape}, Etichette: {labels.shape}")
        #Salvo gli embedding
        emb1 = model(img1)
        emb2 = model(img2)

        #Calcolo la perdita
        loss = criterion(emb1,emb2, labels)

        #BackProp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_corrente += loss.item()
    print(f"Epoch [{epoch+1}], Loss: {loss_corrente/len(train_loader):.4f}")

#validation loop
val_loss = 0.0
correct = 0
total = 0
model.eval()

with torch.no_grad():
    for img1, img2, labels in val_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        val_emb1 = model(img1)
        val_emb2 = model(img2)

        loss = criterion(val_emb1, val_emb2, labels)
        val_loss += loss.item()

        #Calcolo della accuracy
        cosine_similarity = torch.nn.functional.cosine_similarity(val_emb1, val_emb2)
        predictions = (cosine_similarity > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

# Stampa i risultati finali sul validation set
print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {correct/total * 100:.2f}%")        
