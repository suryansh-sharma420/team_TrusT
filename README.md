# Lane Detection Using ENet Model

This project implements a lane detection system using the ENet architecture for semantic segmentation. The code leverages PyTorch for deep learning tasks, and the dataset used is the TuSimple Lane Detection Challenge dataset.

## File Overview

The `lane_enet.ipynb` notebook includes the following components:
1. **Dataset Preprocessing**: Prepares the TuSimple dataset for training and evaluation by resizing images, creating segmentation masks, and generating instance masks.
2. **Model Architecture**: Defines the ENet model, a lightweight deep learning architecture optimized for real-time semantic segmentation.
3. **Training Pipeline**: Implements the training loop, loss computation, and model optimization.
4. **Evaluation**: Tracks the binary segmentation and instance segmentation losses and metrics over epochs.
5. **Visualization**: Plots training metrics such as loss and accuracy for binary segmentation and instance segmentation tasks.

## Key Components

### 1. **Dataset: `LaneDataset`**
The `LaneDataset` class:
- Loads lane detection data from specified dataset paths.
- Resizes images and creates binary segmentation masks and instance masks.
- Returns preprocessed images, segmentation labels, and instance labels for training.

### 2. **Model: `ENet`**
The `ENet` model:
- A lightweight deep learning architecture designed for efficient semantic segmentation.
- Consists of downsampling bottlenecks, regular bottlenecks, and upsampling bottlenecks.
- Handles two branches for binary segmentation and instance embeddings.

### 3. **Loss Function: `DiscriminativeLoss`**
The `DiscriminativeLoss` function:
- Encourages clustering for lane points based on their instance embeddings.
- Penalizes large intra-cluster distances and small inter-cluster distances.

### 4. **Training**
Key features of the training loop:
- CrossEntropyLoss for binary segmentation.
- DiscriminativeLoss for instance segmentation.
- Adam optimizer for model optimization.
- Tracks binary segmentation accuracy and logs metrics to TensorBoard.

### 5. **Visualization**
- Plots binary segmentation loss, instance segmentation loss, and binary segmentation accuracy over training epochs.
- Saves metrics visualization to PNG files.

## How to Use

### 1. Install Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install torch torchvision numpy matplotlib tqdm
```
## 2. Prepare the Dataset
Download the TuSimple Lane Detection Dataset and update the `dataset_path` in the `LaneDataset` class to point to the dataset's location.

## 3. Train the Model
Run the notebook to start training. The training loop will:
- Save metrics to TensorBoard.
- Save the trained model as `lane_detection_model.pth`.

## 4. Evaluate the Model
After training, the saved model can be loaded to make predictions on new lane detection data.

## 5. Visualize Results
The notebook generates visualizations of:
- Binary segmentation loss.
- Instance segmentation loss.
- Binary segmentation accuracy.

## 6. Logs and Model Checkpoints
- Training logs are saved in the `logs` directory.
- The trained model is saved as `lane_detection_model.pth`.

## 📊 Example Output

```text
Epoch 0: Binary Loss = 0.2622, Instance Loss = 2.0723, Binary Accuracy = 0.8780  
Epoch 1: Binary Loss = 0.0647, Instance Loss = 0.3063, Binary Accuracy = 0.9735  
Epoch 2: Binary Loss = 0.0542, Instance Loss = 0.1759, Binary Accuracy = 0.9745

```
## 📈 Model Visualization
Training loss and accuracy plots are automatically generated during the training process and saved as PNG files in the working directory. These visualizations help track model performance across epochs for both binary and instance segmentation tasks.

## ⚙️ Implementation Details

### Model Components:
- **InitialBlock**: The initial convolutional block used for feature extraction from input images.
- **DownsamplingBottleneck**: Reduces the spatial dimensions of the feature maps while increasing the number of channels, useful for compressing input and extracting complex features.
- **UpsamplingBottleneck**: Performs spatial upsampling for decoding purposes, essential for reconstructing the segmentation mask.
- **RegularBottleneck**: Main building block used repeatedly with options for dilated or asymmetric convolutions to enhance receptive fields.

### Loss Components:
- **Binary Segmentation Loss**: Uses `CrossEntropyLoss` to optimize classification between lane and non-lane pixels.
- **Instance Segmentation Loss**: Uses a discriminative loss function to enforce separation between embeddings of different lane instances and compactness within the same lane instance.

## 🔧 Future Improvements
- Increase the number of training epochs to further boost performance and model convergence.
- Fine-tune hyperparameters such as learning rate, batch size, and dropout probability.
- Add evaluation metrics like Intersection over Union (IoU) to better assess the model's segmentation quality.
- Integrate post-processing techniques (e.g., polynomial fitting) to refine detected lane curves.

## 🙏 Acknowledgments
This project is inspired by:
- [TuSimple Lane Detection Challenge](https://github.com/TuSimple/tusimple-benchmark)
- [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)
