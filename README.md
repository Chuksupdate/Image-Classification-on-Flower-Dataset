# Image-Classification-on-Flower-Dataset

# Flower Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) for classifying images of flowers into 5 categories using PyTorch. I leverage a pretrained ResNet-18 model from torchvision, fine-tuned on a custom flower dataset. The model achieves high accuracy on validation and test sets, demonstrating transfer learning for image classification tasks.
The code is provided as a Jupyter notebook (flowers.ipynb) that handles data loading, preprocessing, model training, evaluation, and prediction visualization.
Features

- **Transfer Learning**: Uses pretrained ResNet-18 weights (ImageNet) with a custom classifier head to adapt to 5 flower classes.
- **Data Normalization**: Computes dataset-specific mean and std for better normalization.
- **Train/Validation Split**: 80/20 split from training data, with random seed for reproducibility.
- **Evaluation Metrics**: Includes accuracy, loss tracking, confusion matrix, classification report, and sample predictions.
- **Visualization**: Plots training history and displays test predictions with ground truth vs. predicted labels.
- **Output**: Generates a CSV file (predictions.csv) with test image predictions.

## Dataset

- **Training Data**: ~2,746 images across 5 flower classes (daisy, dandelion, rose, sunflower, tulip) in data/train/.
- **Test Data**: 924 unlabeled images in data/test/ for prediction.
- **Transformation**: Images are RGB-converted if needed and resized to 224x224.

## Requirements

- Python 3.10+
- PyTorch 
- torchvision
- matplotlib
- tqdm
- pandas
- scikit-learn
- Pillow (PIL)


The data directory is structured as follows.

```
project/
├── train/
│   ├── daisy/
│   ├── dandelion/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
└── test/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ... (924 images)
```

## Model Training Pipeline

- Open flowers.ipynb in Jupyter.
- Run cells sequentially:
- Imports and setup (device detection, paths).
- Compute mean/std for normalization.
- Load and split dataset.
- Build and freeze the pretrained ResNet-18 model, replace the final layer for 5 classes.
- Train for 10 epochs (BATCH_SIZE=32, LR=0.001, Adam optimizer, CrossEntropyLoss).
- Evaluate on validation during training.
- Load test images, predict classes, and generate predictions.csv.
- Visualize training curves, confusion matrix, and sample predictions.


## Key Hyperparameters

- Epochs: 10
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: Adam
- Dropout: 0.4 in the classifier
- Normalization: Custom mean [0.5028, 0.4439, 0.3137] and std [0.2437, 0.2187, 0.2224]


## Model Performance

- Validation Accuracy: ~87.6% after 10 epochs.
  ![Validation & Accuracy](/val&acc.png)


## Conclusion
This project has been an invaluable learning experience for me. I gained significant insights into model training and fine-tuning. The model demonstrated impressive performance on the test images, achieving both high precision and high accuracy.

> End.
