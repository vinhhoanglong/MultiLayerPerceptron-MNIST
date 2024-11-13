from MLP import MLP
import torch
import torchvision
import torchvision.transforms as transforms
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])       


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='model/mlp_20241113_224236.save', help='Path to the trained model file')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
args = parser.parse_args()


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

def main(testloader, model_path):
    model = MLP().to(device)  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()   
            
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main(testloader, args.model_path) 