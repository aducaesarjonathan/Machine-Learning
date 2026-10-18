# Handwritten Digit Recognition Using an Artificial Neural Network

## Overview

This project implements a feedforward Artificial Neural Network (ANN) to classify handwritten digits from 0 to 9 using the MNIST dataset.

Developed with Python, TensorFlow, and Keras, the project demonstrates a complete supervised machine learning workflow, including image preprocessing, neural network construction, model training, performance evaluation, prediction visualization, and error analysis.

The work builds on my undergraduate research in artificial neural networks for handwritten digit recognition and reflects my interest in deep learning and computer vision.

## Objectives

- Develop an ANN for handwritten digit classification.
- Apply appropriate image normalization.
- Train and validate a neural network using separate data subsets.
- Evaluate generalization performance on an independent test set.
- Analyze classification errors using a confusion matrix and prediction confidence.
- Visualize model accuracy and save the trained model.

## Dataset

The project uses the MNIST dataset provided through `tf.keras.datasets.mnist`.

| Property | Description |
|---|---|
| Training images | 60,000 |
| Test images | 10,000 |
| Image dimensions | 28 × 28 pixels |
| Image format | Grayscale |
| Number of classes | 10 (digits 0–9) |

A 10% validation subset is taken from the original training data. This produces 54,000 training images and 6,000 validation images, while preserving the 10,000 test images for final evaluation.

## Model Architecture

The model is a fully connected feedforward neural network built using the Keras Sequential API.

| Layer | Configuration |
|---|---|
| Input | 28 × 28 grayscale image |
| Flatten | Converts the image into 784 features |
| Hidden layer 1 | 128 neurons, ReLU |
| Hidden layer 2 | 64 neurons, ReLU |
| Hidden layer 3 | 32 neurons, ReLU |
| Output layer | 10 neurons, Softmax |

ReLU activation functions introduce nonlinearities into the hidden layers. The Softmax output layer generates class probabilities for the ten digit categories.

The predicted digit is the class with the highest probability.

## Methodology

### 1. Data loading and visualization

The MNIST dataset is loaded using TensorFlow. Sample training images are visualized alongside their labels to inspect the input data.

### 2. Image preprocessing

Pixel values are converted to floating-point numbers and scaled to the interval [0, 1] by dividing by 255.

This provides a consistent input scale for neural network training.

### 3. Model construction

A sequential ANN is created with a flattening layer, three dense hidden layers, and a ten-unit Softmax output layer.

### 4. Model compilation

The model uses the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as its primary performance metric.

Sparse categorical cross-entropy is suitable because the target labels are integer class indices rather than one-hot encoded vectors.

### 5. Model training and validation

The network is trained for five epochs with a batch size of 32. Ten percent of the training dataset is reserved for validation.

Training and validation accuracy are recorded at each epoch to monitor learning progress and potential overfitting.

### 6. Final evaluation

After training, the model is evaluated on the independent MNIST test set. Test loss and test accuracy are reported as the primary measures of final classification performance.

The test set is not used for training or validation-based model selection.

### 7. Prediction visualization

Ten randomly selected test images are displayed with their predicted and actual digit labels.

This provides a visual demonstration of the model's predictions.

### 8. Confusion matrix

A confusion matrix compares actual labels with predicted labels across all ten digit classes. It helps reveal which digits are classified correctly and which pairs are frequently confused.

### 9. Error analysis

Misclassified test images are isolated for further examination.

For each incorrect prediction, the analysis compares the highest predicted class probability with the probability assigned to the true class. The five examples with the largest differences are displayed to investigate high-confidence mistakes.

### 10. Model persistence

The trained network is saved in Keras format as `handwritten.model.keras`, allowing it to be loaded later without retraining.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV (imported but not required by the current processing pipeline)

## Installation

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

If you retain the OpenCV import in the script, also install:

```bash
pip install opencv-python
```

## Usage

Run the Python script:

```bash
python handwritten_digit_recognition.py
```

The script loads MNIST, preprocesses the images, trains the ANN, evaluates test performance, displays the visualizations, and saves the trained model.

## Outputs

The project generates:

- Sample MNIST image visualization
- Model architecture summary
- Training and validation accuracy history
- Test loss and accuracy
- Random test-image predictions
- Confusion matrix
- High-confidence error visualizations
- Accuracy graph
- Saved Keras model

Actual performance metrics should be recorded from the execution output rather than assumed in advance.

## Limitations and Future Work

Potential extensions include:

- Comparing the ANN with a convolutional neural network (CNN).
- Experimenting with network depth, width, and regularization.
- Using early stopping and model checkpointing.
- Reporting per-class precision, recall, and F1-score.
- Testing handwritten digits collected outside MNIST.
- Building an interface for interactive digit drawing and prediction.

## Conclusion

This project demonstrates the application of a fully connected artificial neural network to image classification. By combining data preprocessing, validation, independent test evaluation, and error analysis, it provides a foundation for further experiments in deep learning and computer vision.

## Author

**Jonathan Adu-Caesar**

BSc Mathematics, Kwame Nkrumah University of Science and Technology (KNUST).

My research interests include machine learning, deep learning, representation learning, optimization, and natural language processing (NLP).
