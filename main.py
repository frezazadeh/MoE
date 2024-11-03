import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import csv
import config  #Model and Training hyperparameters

# Define the gating function for selecting experts with sparse top-K routing
class GatingFunc(nn.Module):
    def __init__(self, input_dim, num_experts, k=2):
        super(GatingFunc, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.k = k  # number of experts to activate

    def forward(self, x):
        logits = self.fc(x)
        topk_vals, topk_indices = torch.topk(logits, self.k, dim=-1)
        gate_weights = F.softmax(topk_vals, dim=-1)
        sparse_gate_weights = torch.zeros_like(logits).scatter(-1, topk_indices, gate_weights)
        return sparse_gate_weights

# Define each expert as a small feed-forward network
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Define the transformer block with Mixture-of-Experts layer and load-balancing loss
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, output_dim, k=2):
        super(TransformerBlockWithMoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = GatingFunc(input_dim, num_experts, k)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        gate_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        moe_output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        D_i = gate_weights.mean(dim=0)
        load_balancing_loss = (D_i * torch.log(D_i + 1e-8)).sum()
        return self.layer_norm(moe_output + x), load_balancing_loss

# Define the CNN feature extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, feature_dim) # for SVHN, CIFAR10, and CIFAR100
        #self.fc = nn.Linear(128 * 3 * 3, feature_dim) # for MNIST

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Define the CNN + MoE Transformer model
class CNNMoETransformer(nn.Module):
    def __init__(self, input_channels, num_classes, num_experts, hidden_dim, output_dim, k=2):
        super(CNNMoETransformer, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(input_channels=input_channels, feature_dim=output_dim)
        self.transformer_block = TransformerBlockWithMoE(output_dim, num_experts, hidden_dim, output_dim, k)
        self.fc_out = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x, load_balancing_loss = self.transformer_block(x)
        return self.fc_out(x), load_balancing_loss

# Define the Trainer class to manage training and evaluation
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu', save_path='main.csv', alpha=0.001):
        self.model = model.to(device)  # Ensure model is on the specified device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.alpha = alpha
        self.accuracy = Accuracy(task="multiclass", num_classes=config.NUM_CLASSES).to(device)
        self.precision = Precision(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.recall = Recall(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.f1 = F1Score(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.metrics_history = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            total_acc, total_prec, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, load_balancing_loss = self.model(images)
                loss = self.criterion(outputs, labels) + self.alpha * load_balancing_loss
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                total_acc += self.accuracy(outputs, labels)
                total_prec += self.precision(outputs, labels)
                total_recall += self.recall(outputs, labels)
                total_f1 += self.f1(outputs, labels)

            # Compute epoch metrics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = total_acc / len(self.train_loader)
            epoch_prec = total_prec / len(self.train_loader)
            epoch_recall = total_recall / len(self.train_loader)
            epoch_f1 = total_f1 / len(self.train_loader)

            # Validation phase
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs, _ = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)

            # Append metrics for this epoch
            self.metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'accuracy': epoch_acc.item(),
                'precision': epoch_prec.item(),
                'recall': epoch_recall.item(),
                'f1_score': epoch_f1.item()
            })

            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Accuracy: {epoch_acc:.4f}, Precision: {epoch_prec:.4f}, "
                  f"Recall: {epoch_recall:.4f}, F1 Score: {epoch_f1:.4f}")

        print("Training complete.")
        self.save_metrics()

    def save_metrics(self):
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_history)
        print(f"Metrics saved to {self.save_path}.")

# Main script for running the training process
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNMoETransformer(
        input_channels=config.INPUT_CHANNELS, 
        num_classes=config.NUM_CLASSES, 
        num_experts=config.NUM_EXPERTS, 
        hidden_dim=config.HIDDEN_DIM, 
        output_dim=config.OUTPUT_DIM,
        k=config.TOP_K
    ).to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if config.DATASET == 'CIFAR10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif config.DATASET == 'CIFAR100':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        val_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        
    elif config.DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    elif config.DATASET == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        train_data = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        val_data = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, epochs=config.EPOCHS, device=device)
    trainer.train()

