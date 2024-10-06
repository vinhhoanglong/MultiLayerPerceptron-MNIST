from MLP import MLP
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os




transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])       
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

def main(testloader, model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()   
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/mlp_20241006_220807.save')
    args = parser.parse_args()
    main(testloader, args.model_path) 

