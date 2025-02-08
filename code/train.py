import random
import torch
from torch.utils.data import DataLoader, Subset
from ResNet import ResNet, resnet18
from loss import ElasticFaceLoss, ContrastiveLoss
from dataset import LFWDataLoader
import torch.optim as optim 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#DataLoader
#Training set 
train_data = LFWDataLoader(
    root_dir='/Users/riccardofantechi/Desktop/Universita/Quarto anno/Deep Learning/ElasticFace_IMPL/DataSet/archive/lfw-deepfunneled/lfw-deepfunneled',
    match_file='/Users/riccardofantechi/Desktop/Universita/Quarto anno/Deep Learning/ElasticFace_IMPL/DataSet/archive/matchpairsDevTrain.csv',
    mismatch_file='/Users/riccardofantechi/Desktop/Universita/Quarto anno/Deep Learning/ElasticFace_IMPL/DataSet/archive/mismatchpairsDevTrain.csv'
)

match_indices = list(range(1100))
mismatch_indices = list(range(1100,2200))

random.shuffle(match_indices)
random.shuffle(mismatch_indices)

subset_size = 1100
subset_indices = match_indices[:subset_size // 2] + mismatch_indices[:subset_size // 2]

balanced_subset = Subset(train_data, subset_indices)
train_loader = DataLoader(balanced_subset, batch_size = 32, shuffle = True, num_workers=0)

#Validation set
val_data = LFWDataLoader(
    root_dir='/Users/riccardofantechi/Desktop/Universita/Quarto anno/Deep Learning/ElasticFace_IMPL/DataSet/archive/lfw-deepfunneled/lfw-deepfunneled',
    match_file='/Users/riccardofantechi/Desktop/Universita/Quarto anno/Deep Learning/ElasticFace_IMPL/DataSet/archive/matchpairsDevTest.csv',
    mismatch_file='/Users/riccardofantechi/Desktop/Universita/Quarto anno/Deep Learning/ElasticFace_IMPL/DataSet/archive/mismatchpairsDevTest.csv'
)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
model = resnet18(num_classes=512).to(device)

#Loss e opt
criterion = ElasticFaceLoss(s = 128.0, m = 0.5, std = 0.05)
#criterion = ContrastiveLoss()

optimizer = optim.Adam(model.parameters(),lr = 0.001)

#Training loop
for epoch in range(20):
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