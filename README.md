# Machine Learning using Jupiter Notebook and Google Colab

**Models and Algorithms:**
  1. [**_k Nearest Neighbor_**](https://github.com/ElizaLo/ML-with-Jupiter#k-nearest-neighbor)
  2. [**_Linear Regression_**](https://github.com/ElizaLo/ML-with-Jupiter#linear-regression)
  3. [**_Logistic Regression_**](https://github.com/ElizaLo/ML-with-Jupiter#logistic-regression)
  4. [**_Fully Connected Neural Networks_**](https://github.com/ElizaLo/ML-with-Jupiter#fully-connected-neural-networks)
  5. [**_Computer Vision - Blurred Images_**](https://github.com/ElizaLo/ML-with-Jupiter#computer-vision---blurred-images)
  6. [**_OpenCV_**](https://github.com/ElizaLo/ML-with-Jupiter#opencv)
  7. [**_Convolutional Neural Network (CNN)_**](https://github.com/ElizaLo/ML-with-Jupiter#convolutional-neural-network-cnn)
  
**Projects:**
  - [**_Spam Detection_**](https://github.com/ElizaLo/ML-with-Jupiter#spam-detection)
  - [**_Text Generator_**](https://github.com/ElizaLo/ML-with-Jupiter#text-generator)
 
## What's is the difference between _train, validation and test set_, in neural networks?

**Training Set:** this data set is used to adjust the weights on the neural network.

**Validation Set:** this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over the validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.

**Testing Set:** this data set is used only for testing the final solution in order to confirm the actual predictive power of the network.
 
# **Models and Algorithms:**

 1. ## **k Nearest Neighbor**
    - [Code](https://github.com/ElizaLo/ML/blob/master/k%20Nearest%20Neighbor/kNN.py)
    - [Subset of MNIST](https://pjreddie.com/projects/mnist-in-csv/)

 2. ## **Linear Regression**
    - [Code](https://github.com/ElizaLo/ML-with-Jupiter/tree/master/P2)
    
 3. ## **Logistic Regression**
    - [Code](https://github.com/ElizaLo/ML-with-Jupiter/tree/master/P3)
 
 4. ## **Fully Connected Neural Networks**
    - Fully connected neural network that recognizes handwriting numbers from  MNIST database (Modified National Institute of     Standards and Technology database)
    - [MNIST Database](https://pjreddie.com/projects/mnist-in-csv/)
    - [Code](https://github.com/ElizaLo/ML-with-Jupiter/tree/master/P4)
    
 5. ## **Computer Vision - Blurred Images**
    - Algorithms for restoring blurred images
    - Using **_scipy_** and **_imageio_**
    - [Code](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/P5/ML_Practice_5.ipynb)
 
 6. ## **OpenCV**
    - Face Detector on WebCam
    - Camera Recording into file
    - Detecting blue squares from WebCam
    - [OpenCV Documentation](https://opencv-python-tutroals.readthedocs.io/en/latest/)
    - [Code](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/P6/ML_Practice_6.ipynb)
    
 7. ## **Convolutional Neural Network (CNN)**
    - [Code](https://github.com/ElizaLo/ML-with-Jupiter/tree/master/P7)
 
 
# Projects: 

 - # **Spam Detection**
 
   Methods:
    - **_Naive Bayes spam filtering_**
    - **_K-Nearest Neighbors algorithm_**
    - **_Decision Tree learning_**
    - **_Support Vector Machine (SVM)_**
    - **_Random Forest_**
    
  - [Code](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Spam%20Detection/Spam_Detection.ipynb)
  - [SMS Spam Collection Dataset](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Spam%20Detection/spam.csv)
  
 - # **Text Generator**
   
   Neural Network for generating text based on training txt file using **_Google Colab_**. 
   As a base text were used **_Alice in Wonderland_** by Lewis Carroll.
   
  - [Code](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Text%20Generator%20/%20Text_Generator.ipynb)
  - [Base text - **Alice in Wonderland**](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Text%20Generator/alice_in_wonderland.txt)
  - [Formatted text of **Alice in Wonderland**](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Text%20Generator/alice_formatted.txt)
