from MLP import MLP
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP().to(device) 
criterion = nn.CrossEntropyLoss()

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD', help='Optimizer to use: SGD or Adam')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])       
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)  

def main(model, trainloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), f'model/mlp_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.save')


if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

main(model, trainloader, criterion, optimizer, args.num_epochs)