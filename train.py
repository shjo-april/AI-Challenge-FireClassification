import tqdm
import torch
import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from utils import Dataset, Network

if __name__ == '__main__':
    train_dataset = Dataset('./train.xlsx', 'train')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=1)

    model = Network(num_classes=2)
    
    model.cuda()
    model.train()

    loss_fn = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=4e-5)

    max_epochs = 20

    for epoch in range(1, max_epochs+1):
        losses = []

        for images, labels in tqdm.tqdm(train_loader, desc=f'Epoch: {epoch:03d}'):
            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)

            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(losses)
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.04f}')

        torch.save(model.state_dict(), 'cnn.pt')