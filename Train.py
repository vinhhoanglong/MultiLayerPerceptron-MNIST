from MLP import MLP
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import datetime

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])       
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


def main(model, trainloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), f'model/mlp_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.save')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5)
    args = parser.parse_args()
    main(model, trainloader, criterion, optimizer, args.num_epochs)

