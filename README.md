# Deep-learning-HCC-VETC

# Deep Learning Model for Radiomics and Pathomics

This project uses ResNet50, Densenet121, Vision Transformer, and Swin Transformer to construct deep radiomics and pathomics models via transfer learning for 2D image classification.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- pandas
- scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```bash
    pip install torch torchvision numpy matplotlib tqdm pandas scikit-learn
    ```

## Usage

1. Set the paths for the training and testing datasets:
    ```python
    train_dir = r'path_to_train_directory'
    test_dir = r'path_to_test_directory'
    ```

2. Run the training script:
    ```bash
    python deep_learning_model_code.py
    ```

## Model Training

The script trains the model using the specified architecture (ResNet50, Densenet121, Vision Transformer, or Swin Transformer) with transfer learning. The training process includes data augmentation, model setup, and training with a learning rate scheduler.

## Saving the Model

The trained model is saved to `my_model.pth`:
    ```python
    PATH = './my_model.pth'
    torch.save(model.state_dict(), PATH)
    ```

