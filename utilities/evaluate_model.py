import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

def evaluate_model(model_path, data_dir, transform, device):
    model = torch.load(model_path, map_location=device)
    model.eval()

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    evaluate_model("best_model.pth", "FER2013_processed/test", transform, 'cuda')
