import csv
import matplotlib.pyplot as plt

# Load metrics from CSV file
def load_metrics(file_path):
    epochs, train_loss, val_loss, accuracy, precision, recall, f1_score = [], [], [], [], [], [], []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            accuracy.append(float(row['accuracy']))
            precision.append(float(row['precision']))
            recall.append(float(row['recall']))
            f1_score.append(float(row['f1_score']))
    return epochs, train_loss, val_loss, accuracy, precision, recall, f1_score

# Plot metrics and save the figure
def plot_metrics(epochs, train_loss, val_loss, accuracy, precision, recall, f1_score, save_path=None):
    plt.figure(figsize=(10, 8))

    # Loss Plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy, Precision, Recall, and F1 Score Plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.plot(epochs, precision, label='Precision')
    plt.plot(epochs, recall, label='Recall')
    plt.plot(epochs, f1_score, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Training Metrics for CIFAR100 Dataset with Expert= 21  and Top-k = 14')
    plt.legend()

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300,bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()

# Load and plot metrics
file_path = ''
save_path = 'metrics_CNN_MoE.png'  # File path to save the plot
epochs, train_loss, val_loss, accuracy, precision, recall, f1_score = load_metrics(file_path)
print(f1_score)
plot_metrics(epochs, train_loss, val_loss, accuracy, precision, recall, f1_score, save_path)
