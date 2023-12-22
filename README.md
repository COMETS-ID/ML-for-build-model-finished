**In this project, we developed one machine learning model that plays an important role in the application. In this project, we use the CNN model. Convolutional Neural Networks (CNNs) are a type of deep learning model that has proven to be very effective in computer vision tasks, including emotion recognition. Our data is processed and divided into categorical features. Then the Sequential model with several Dense layers is used to predict labels. This model was trained using Adam optimizer and categorical_crossentropy for 50 epochs. We also use early stopping. Early stopping allows training to be stopped early if there is no significant improvement in performance on the validation set. This helps prevent overfitting.**


## Here's a breakdown of how CNNs work in the context of emotion recognition:
- Convolutional Layers:
CNNs use convolutional layers to scan the input data (in this case, images) using filters or kernels. These filters are small windows that slide over the input image and perform element-wise multiplication with the local regions of the image.
- Pooling Layers:
After convolutional layers, pooling layers are often employed to reduce the spatial dimensions of the obtained feature maps while retaining the most important information. Max pooling is a common technique, where the maximum value within a small window is retained, and the rest are discarded.
- Fully Connected Layers:
Following the convolutional and pooling layers, fully connected layers are used to connect every neuron in one layer to every neuron in the next layer. These layers learn global patterns and relationships in the data.
- Emotion Recognition:
For emotion recognition, CNNs are trained on a dataset containing images labeled with different emotions (e.g., happy, sad, angry). The network learns to extract hierarchical features from the input images, capturing both low-level features like edges and textures, and high-level features such as facial expressions.
In this project, we have 7 labels that are angry, sad, happy, neutral, fear, surprise, disgust
- Training:
The model is trained using a labeled dataset through a process called backpropagation. During training, the network adjusts its weights to minimize the difference between its predicted emotions and the actual labels in the training data.
- Activation Functions:
Activation functions, such as Rectified Linear Unit (ReLU), are used to introduce non-linearity to the model. This enables the network to learn complex relationships and make the model more expressive.  In this project, we use ReLu and Softmax. The softmax function is used in the output layer in multi-class classification problems
- Output Layer:
The output layer typically has neurons corresponding to different emotion classes (e.g., happy, sad, angry). The softmax activation function is commonly used to convert the network's raw output into probabilities, indicating the likelihood of each emotion.
- Loss Function:
The loss function measures the difference between the predicted emotions and the actual labels. The goal during training is to minimize this loss.

## Technology and Tools Used
- Python: The programming language used in the implementation of recommendation and price prediction models.
- Pandas: A Python library used to manipulate and analyze data.
- NumPy: A Python library used for math operations and array training.
- Matplotlib: Python libraries used for data visualization, such as creating plots, histograms, and counterplots.
- TensorFlow and Keras: Machine learning frameworks used to build and train neural network models.

## Our Dataset Link
[Dataset](https://bit.ly/COMETS_dataset)https://bit.ly/COMETS_dataset)

## Requirements
-Make sure you have install any Python version
- install all libraries:
  - import pandas as pd
  - import os
  - import glob as gb
  - from tensorflow import kerasfrom keras.models import Sequential
  - from keras.layers import Conv2D
  - from keras.layers import MaxPooling2D
  - from keras.layers import Flatten
  - from keras.layers import Dense
  - from keras.preprocessing.image import ImageDataGenerator
  - import random
  - import matplotlib.pyplot as plt
  - import matplotlib.image as mpimg

 ## Our model Performance
 








