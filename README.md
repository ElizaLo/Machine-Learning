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
  - **_Quora Insincere Questions Classification_**
  - [**_Question Answering System using BiDAF Model on SQuAD_**](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD)
  
**Natural Language Processing:**  
- [ ] [**NLP Projects Repository**](https://github.com/ElizaLo/NLP)
 
## What's is the difference between _train, validation and test set_, in neural networks?

**Training Set:** this data set is used to adjust the weights on the neural network.

**Validation Set:** this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over the validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.

> The **validation data set** is a set of data for the function you want to learn, which you are not directly using to train the network. You are training the network with a set of data which you call the training data set. If you are using gradient based algorithm to train the network then the error surface and the gradient at some point will completely depend on the training data set thus the training data set is being directly used to adjust the weights. To make sure you don't overfit the network you need to input the validation dataset to the network and check if the error is within some range. Because the validation set is not being using directly to adjust the weights of the network, therefore a good error for the validation and also the test set indicates that the network predicts well for the train set examples, also it is expected to perform well when new example are presented to the network which was not used in the training process.

**Testing Set:** this data set is used only for testing the final solution in order to confirm the actual predictive power of the network.

**Also**, in the case you do not have enough data for a validation set, you can use **cross-validation** to tune the parameters as well as estimate the test error.

**Cross-validation set** is used for model selection, for example, select the polynomial model with the least amount of errors for a given parameter set. The test set is then used to report the generalization error on the selected model. 

**[Early stopping](https://en.wikipedia.org/wiki/Early_stopping)** is a way to stop training. There are different variations available, the main outline is, both the train and the validation set errors are monitored, the train error decreases at each iteration ([backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and brothers) and at first the validation error decreases. The training is stopped at the moment the validation error starts to rise. The weight configuration at this point indicates a model, which predicts the training data well, as well as the data which is not seen by the network . But because the validation data actually affects the weight configuration indirectly to select the weight configuration. This is where the Test set comes in. This set of data is never used in the training process. Once a model is selected based on the validation set, the test set data is applied on the network model and the error for this set is found. This error is a representative of the error which we can expect from absolutely new data for the same problem.
 
# **Models and Algorithms:**

 1. ## **k Nearest Neighbor**
    - [Code](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/k%20Nearest%20Neighbor/kNN.py)
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
   
  - [Code](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/Text%20Generator/Text_Generator.ipynb)
  - [Base text - **Alice in Wonderland**](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Text%20Generator/alice_in_wonderland.txt)
  - [Formatted text of **Alice in Wonderland**](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Text%20Generator/alice_formatted.txt)
