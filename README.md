# Image-Processing-Project
Aygaz Görüntü İşleme Bootcamp: Yeni Nesil Proje Kampı

# Animal Classification using CNNs
##Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify images of animals into 10 distinct classes. The dataset used for this task is 'Animals with Attributes 2', sourced from Kaggle. The aim is to develop a robust image classification model while applying techniques like data augmentation, brightness manipulation, and color constancy algorithms to analyze model performance under different conditions.


##Dataset
The dataset consists of images of 10 animal classes:

Collie, Dolphin, Elephant, Fox, Moose, Rabbit, Sheep, Squirrel, Giant Panda, Polar Bear.
Class Distribution: Each class contains 650 images.
Preprocessing: All images were resized to 128x128 pixels and normalized (values scaled between 0 and 1).


##Libraries Used
The following libraries and frameworks were used to build and analyze the project:

TensorFlow/Keras: For designing and training the CNN model.
OpenCV: For brightness manipulation and image processing.
NumPy: For numerical operations and dataset handling.
Matplotlib: For visualizing training progress and results.
scikit-learn: For dataset splitting and evaluation metrics.


## Methodology
### 1. CNN Architecture
A CNN model was built using the following approach:

Base Model: Pre-trained VGG16 architecture (transfer learning).
Custom Layers:
Flatten layer.
Dense layers with ReLU activation for feature extraction.
Dropout layers to prevent overfitting.
Final Dense layer with Softmax activation for classification into 10 classes.


### 2. Data Augmentation
To increase dataset diversity, the following augmentations were applied:

Random rotations up to 20°.
Random width/height shifts up to 10%.
Random horizontal flips.


### 3. Brightness Manipulation
A function was applied to increase the brightness of test images to evaluate the model's robustness. Brightness manipulation was done using OpenCV with the formula:

python
Kodu kopyala
manipulated = cv2.convertScaleAbs(img, alpha=1.5, beta=30)


### 4. Color Constancy Algorithm
The Gray World Algorithm was applied to correct the color balance of manipulated images and analyze its impact on model performance.

### 5. Evaluation Metrics
The model was evaluated using accuracy on three test sets:
Original Test Set
Brightness Manipulated Test Set
Color Corrected Test Set

## Results
### Test Set	Accuracy
Original Test Set	35.49%
Brightness Manipulated Test Set	9.69%
Color Corrected Test Set	9.69%
