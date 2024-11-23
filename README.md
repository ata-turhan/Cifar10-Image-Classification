## CIFAR-10 Image Classification with Convolutional Neural Networks

Welcome to this Kaggle notebook repository for CIFAR-10 image classification. In this project, we build a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes. This project is structured to guide you through each step of the machine learning pipeline, from data extraction to model training and evaluation, culminating in a submission for Kaggle evaluation.

### Repository Structure

```
.
├── .gitignore                        # Specifies files and directories to be ignored by Git
├── LICENSE                           # License file (MIT License)
├── README.md                         # Project overview and setup instructions
├── cifar10-image-classification.ipynb # Kaggle notebook with all steps, including training and evaluation
```

### Notebook Overview

This notebook is organized into several key tasks to create a robust image classification model:

1. **Introduction**: Overview of the CIFAR-10 dataset and project objectives.
2. **Importing Libraries**: Importing essential libraries for data manipulation, model building, and training.
3. **Loading the Dataset**: Extracting CIFAR-10 training and test datasets from compressed `.7z` files.
4. **Data Preprocessing**: Loading, resizing, normalizing the images, and splitting the dataset into training and validation sets.
5. **Data Augmentation**: Applying transformations to training images to improve model generalization.
6. **Building the Model**: Designing a CNN using TensorFlow/Keras, complete with L2 regularization, dropout layers, and batch normalization.
7. **Training the Model**: Training the CNN model using augmented data, early stopping, and learning rate reduction.
8. **Plotting Accuracy and Loss**: Visualizing training and validation metrics for accuracy and loss to evaluate model performance.
9. **Making Predictions**: Using the trained model to generate predictions on the test data and preparing the submission file.

### Getting Started

Follow these steps to set up and run the project on your local machine:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ata-turhan/cifar10-image-classification.git
   cd cifar10-image-classification
   ```

2. **Install Required Dependencies**
   Install the necessary Python packages listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**
   - **Download** the CIFAR-10 dataset from Kaggle.
   - **Extract** the dataset files (`train.7z` and `test.7z`) into the correct folders using the extraction function provided.

4. **Run the Notebook**
   - You can open the `cifar10-image-classification.ipynb` notebook and execute the cells step-by-step to train the model and make predictions.

### Key Features
- **Data Augmentation**: Includes rotation, shifting, zooming, and flipping to enhance model robustness.
- **CNN Architecture**: Built with several convolutional blocks, batch normalization, and dropout for effective learning.
- **Regularization**: L2 regularization and dropout are applied to reduce overfitting.
- **Callbacks**: Early stopping and learning rate reduction ensure efficient training.
- **Kaggle Submission**: The final predictions are generated and saved in a format compatible with Kaggle submission requirements.

### Results
- The model achieves reasonable performance by training on augmented data, with accuracy and loss plots provided for analysis.
- A submission file (`submission.csv`) is generated for evaluation on the Kaggle leaderboard.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- **Kaggle**: For providing the CIFAR-10 dataset.
- **TensorFlow/Keras**: For their easy-to-use deep learning framework.

### Contact
For any questions or suggestions, please reach out:
- **GitHub**: [ata-turhan](https://github.com/ata-turhan)
- **LinkedIn**: [ataturhan](https://www.linkedin.com/in/ataturhan/)

We hope this notebook and repository help you learn and implement a deep learning solution for image classification using CIFAR-10. Feel free to fork, modify, and experiment!

