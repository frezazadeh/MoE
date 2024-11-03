# Model hyperparameters
DATASET = 'CIFAR100'  # Options: 'SVHN', 'CIFAR100', 'CIFAR10', 'MNIST'
INPUT_DIM = 32 * 32 * 3 if DATASET in ['SVHN', 'CIFAR10', 'CIFAR100'] else 28 * 28  # Adjust input dimensions
INPUT_CHANNELS = 3 if DATASET in ['SVHN', 'CIFAR10', 'CIFAR100'] else 1  # 3 for SVHN, CIFAR-10, and CIFAR-100; 1 for MNIST
NUM_CLASSES = 10 if DATASET in ['SVHN', 'CIFAR10'] else 100 if DATASET == 'CIFAR100' else 10  # 10 for SVHN and CIFAR-10, 100 for CIFAR-100, 10 for MNIST
NUM_EXPERTS = 5 # Number of experts in MoE
HIDDEN_DIM = 128  # Hidden dimension for each expert
OUTPUT_DIM = 256  # Output dimension after feature extraction
TOP_K = 2  # Number of experts to activate in MoE

# Training hyperparameters
BATCH_SIZE = 64  # Batch size for training and validation
LEARNING_RATE = 0.001  # Learning rate for the optimizer
EPOCHS = 150  # Number of training epochs
