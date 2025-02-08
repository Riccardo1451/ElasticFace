import torch
import torch.nn as nn
import torch.nn.functional as F


class ElasticFaceLoss(nn.Module):
    def __init__(self, s = 64.0, m = 0.35, std = 0.0125):
        super(ElasticFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.std = std

    def forward(self, emb_1, emb_2, labels):
        #Clacoliamo il coseno della distanza dei due embedding
        cosine_distance = F.cosine_similarity(emb_1,emb_2)

        #Calcolo del margine elastico
        margin = torch.normal(mean=self.m, std = self.std, size = (cosine_distance.size(0),), device = emb_1.device)

        #Calcolo del nuovo valore
        cosine_distance_margin = cosine_distance - labels * margin

        # Applica la funzione logistica per la classificazione binaria
        loss = F.binary_cross_entropy_with_logits(self.s * cosine_distance_margin, labels)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calcola la distanza euclidea tra i due embedding
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Perdita contrastiva
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss