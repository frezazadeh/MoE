import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import csv

# Import hyperparameters from config
import config

# Define the gating Function for selecting experts with sparse top-K routing
class GatingFunc(nn.Module):
    def __init__(self, input_dim, num_experts, k=2):
        super(GatingFunc, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.k = k  # number of experts to activate

    def forward(self, x):
        logits = self.fc(x)
        # Select top-k values and their indices along the expert dimension (dim=-1)
        topk_vals, topk_indices = torch.topk(logits, self.k, dim=-1)
        
        # Apply softmax to top-k values only to get the gate weights
        gate_weights = F.softmax(topk_vals, dim=-1)
        
        # Create a sparse mask for the top-k experts, then scale by gate_weights
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
        self.num_experts = num_experts

    def forward(self, x):
        # Gating mechanism: select experts based on input features
        gate_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        moe_output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        # Compute auxiliary load-balancing loss
        D_i = gate_weights.mean(dim=0)  # Average gating probability per expert
        load_balancing_loss = (D_i * torch.log(D_i + 1e-8)).sum()  # Minimize entropy of D_i
        #load_balancing_loss = torch.clamp(load_balancing_loss, min=0)  # Clamp to avoid negative values

        return self.layer_norm(moe_output + x), load_balancing_loss  # Return output and balancing loss


# Define a simple transformer for images with MoE layers
class MoETransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts, hidden_dim, output_dim, k=2):
        super(MoETransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)  # Add input projection layer
        self.transformer_block = TransformerBlockWithMoE(output_dim, num_experts, hidden_dim, output_dim, k)
        self.fc_out = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.input_projection(x)  # Project input to match output_dim
        x, load_balancing_loss = self.transformer_block(x)
        return self.fc_out(x), load_balancing_loss  # Return logits and auxiliary loss

# Define the Trainer class to manage training and evaluation
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu', save_path='metrics.csv', alpha=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.alpha = alpha  # Coefficient for load-balancing loss
        # Initialize metrics with task argument for multiclass classification
        self.accuracy = Accuracy(task="multiclass", num_classes=config.NUM_CLASSES).to(device)
        self.precision = Precision(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.recall = Recall(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.f1 = F1Score(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        # Lists to store metrics per epoch
        self.metrics_history = []

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            total_acc, total_prec, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0
            for images, labels in self.train_loader:
                images = images.view(images.size(0), -1).to(self.device)  # Flatten images
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, load_balancing_loss = self.model(images)
                loss = self.criterion(outputs, labels) + self.alpha * load_balancing_loss  # Combine with auxiliary loss
                
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track running loss
                running_loss += loss.item()
                
                # Calculate metrics
                total_acc += self.accuracy(outputs, labels)
                total_prec += self.precision(outputs, labels)
                total_recall += self.recall(outputs, labels)
                total_f1 += self.f1(outputs, labels)

            # Average training metrics for the epoch
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = total_acc / len(self.train_loader)
            epoch_prec = total_prec / len(self.train_loader)
            epoch_recall = total_recall / len(self.train_loader)
            epoch_f1 = total_f1 / len(self.train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.view(images.size(0), -1).to(self.device)
                    labels = labels.to(self.device)
                    outputs, _ = self.model(images)  # No load-balancing loss in validation
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)

            # Save metrics to history
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
        # Save the metrics history to a CSV file
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_history)
        print(f"Metrics saved to {self.save_path}.")

if __name__ == "__main__":
    # Model, loss, optimizer setup
    model = MoETransformer(
        input_dim=config.INPUT_DIM, 
        num_classes=config.NUM_CLASSES, 
        num_experts=config.NUM_EXPERTS, 
        hidden_dim=config.HIDDEN_DIM, 
        output_dim=config.OUTPUT_DIM,
        k=config.TOP_K
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Data loading and processing based on dataset choice in config
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

elif config.DATASET == 'MNIST':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)

# Initialize and run the trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, epochs=config.EPOCHS, device=device)
trainer.train()
