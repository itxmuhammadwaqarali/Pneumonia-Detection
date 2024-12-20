# Pneumonia Detection Task

## Overview

This project focuses on the detection of pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and Transfer Learning. The primary objective was to develop a robust and efficient model capable of classifying chest X-rays as 'Normal' or 'Pneumonia'.

## Key Steps

### 1. **Data Acquisition**

- **Dataset Source**: Downloaded from Kaggle using the `chest-xray-pneumonia` dataset.
- The dataset was organized into `train`, `test`, and `validation` sets, each containing two subfolders for `NORMAL` and `PNEUMONIA` images.

### 2. **Data Preprocessing**

- **Image Resizing**: Resized all images to 224x224 pixels to maintain consistency with common pre-trained model input sizes.
- **Normalization**: Normalized pixel values to the range [0, 1] to aid in training convergence.
- **Data Augmentation**: Applied augmentation techniques like rotation, flipping, brightness adjustments, and zoom to increase dataset variability and reduce overfitting.

### 3. **Data Splitting**

- Divided the dataset into training, validation, and testing sets to ensure proper evaluation of the model.

### 4. **Model Development**

- **Custom CNN**: Built a custom CNN with multiple convolutional, max-pooling, and dropout layers to extract spatial features from images.
- **Transfer Learning**: Fine-tuned pre-trained models like `EfficientNetB0` and `EfficientNetB3` to leverage feature extraction from large ImageNet-trained networks.
- **Activation Functions**: Used ReLU activation for hidden layers and sigmoid activation for the binary classification output.
- **Optimization**: Utilized the Adam optimizer with a binary cross-entropy loss function.

### 5. **Model Training**

- Trained the models using the prepared training set, validating the performance on the validation set.
- Applied an early stopping mechanism to halt training if no improvements were observed in validation loss for 15 consecutive epochs.

### 6. **Evaluation and Metrics**

- **Accuracy**: Measured the accuracy of the model on the validation and test sets.
- **Loss**: Visualized the training and validation loss over epochs to detect overfitting.
- **ROC-AUC**: Evaluated the Area Under the Receiver Operating Characteristic Curve to assess classification performance.
- **Confusion Matrix**: Computed the confusion matrix to measure sensitivity, specificity, precision, and recall.

### 7. **Results Comparison**

- **Augmented Data vs Non-Augmented Data**: Compared the performance of the model trained on augmented data against non-augmented data.
- Observed significant improvements in model generalization due to the augmentation techniques applied.

### 8. **Results**

- **Final Accuracy**: The model achieved a testing accuracy of over 85%.
- **Test Loss**: The loss during testing was within an acceptable range, indicating no significant overfitting.

### 9. **Future Enhancements**

- **Hyperparameter Tuning**: Further tuning of learning rates, batch sizes, and optimizer selection.
- **Ensemble Models**: Combine predictions from multiple models to increase robustness.
- **Explainability**: Use Grad-CAM or similar methods to visualize important regions in X-ray images for better interpretability.

### 10. **Tools and Libraries Used**

- **Languages**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Plotly, and Scikit-learn.

This project demonstrated the effectiveness of CNNs and Transfer Learning for medical image classification tasks, achieving strong performance on the Pneumonia Detection task.

