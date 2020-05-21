<img src="https://github.com/ElizaLo/Machine-Learning/blob/master/images/Banner_Machine_Learning.png" width="900" height="100">

This repository contains examples of popular machine learning algorithms implemented in Python with mathematics behind them being explained.


> - [ ] For **Octave/MatLab** version of Machine Learning algorithms please check [Machine Learning Course in Octave / MatLab](https://github.com/ElizaLo/Machine-Learning-Course-Octave) repository.

> - [ ] For **Deep Learning** algorithms please check [Deep Learning](https://github.com/ElizaLo/Deep-Learning) repository.

> - [ ] For **Natural Language Processing** (NLU = NLP + NLG) please check [Natural Language Processing](https://github.com/ElizaLo/NLP-Natural-Language-Processing) repository.

> - [ ] For **Computer Vision** please check [Computer Vision](https://github.com/ElizaLo/Computer-Vision) repository.


## Table of Contents

- [Machine Learning Map](https://github.com/ElizaLo/Machine-Learning#machine-learning-map)
- [Top University Courses](https://github.com/ElizaLo/Machine-Learning#-top-university-courses)
- [Coursera Courses](https://github.com/ElizaLo/Machine-Learning#-coursera-courses)
- [YouTube Videos](https://github.com/ElizaLo/Machine-Learning#-youtube-videos)
- [Books](https://github.com/ElizaLo/Machine-Learning#-books)
- [Websites](https://github.com/ElizaLo/Machine-Learning#%EF%B8%8F-websites)
- [Other](https://github.com/ElizaLo/Machine-Learning#-other)
- [Reinforcement Learning ](https://github.com/ElizaLo/Machine-Learning#reinforcement-learning)
  - Books
  - Classes
  - Task / Tutorials
  - GitHub Repositories
- [Datasets](https://github.com/ElizaLo/Machine-Learning#-datasets-list)
- Implementation
  - [Supervised Learning](https://github.com/ElizaLo/Machine-Learning/tree/master/Supervised%20Learning)
    - Classification
      - [K Nearest Neighbor, K-NN](https://github.com/ElizaLo/ML-with-Jupiter#k-nearest-neighbor)
      - [Logistic Regression](https://github.com/ElizaLo/ML-with-Jupiter#logistic-regression)
    - Regression
      - [Linear Regression](https://github.com/ElizaLo/ML-with-Jupiter#linear-regression)
      - [Polynomial Regression](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#polynomial-regression)
  - [Unsupervised Learning](https://github.com/ElizaLo/Machine-Learning/tree/master/Unsupervised%20Learning)
- [What's is the difference between train, validation and test set, in neural networks?](https://github.com/ElizaLo/Machine-Learning#whats-is-the-difference-between-train-validation-and-test-set-in-neural-networks)
- [Models and Algorithms Implementation](https://github.com/ElizaLo/Machine-Learning#%EF%B8%8F-models-and-algorithms)
  4. [Connected Neural Networks](https://github.com/ElizaLo/ML-with-Jupiter#fully-connected-neural-networks)
  5. [Computer Vision - Blurred Images](https://github.com/ElizaLo/ML-with-Jupiter#computer-vision---blurred-images)
  6. [OpenCV](https://github.com/ElizaLo/ML-with-Jupiter#opencv)
  7. [Convolutional Neural Network (CNN)](https://github.com/ElizaLo/ML-with-Jupiter#convolutional-neural-network-cnn)
  8. [Gated Recurrent Units (GRU)](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab#gated-recurrent-units-gru)
- Projects
  - [Spam Detection](https://github.com/ElizaLo/ML-with-Jupiter#spam-detection)
  - [Text Generator](https://github.com/ElizaLo/ML-with-Jupiter#text-generator)
  - **_Quora Insincere Questions Classification_**
  - [**_Question Answering System using BiDAF Model on SQuAD_**](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD)

    
## Machine Learning Map
![machine-learning-map.png](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/images/machine-learning-map.png)


## 🎓 Top University Courses 

- [ ] [Курс Машинное обучение. Воронцов](https://www.youtube.com/playlist?list=PLww5O9qI8iPP-mZf8mCMff3eMWFBHr0m1)
- [ ] [Курс "Машинное обучение" на ФКН ВШЭ, Евгений Соколов](https://github.com/esokolov/ml-course-hse)
- [ ] [Data Mining](https://www.cs.utah.edu/~jeffp/teaching/cs5955.html)

**Introductory Lectures:**

These are great courses to get started in machine learning and AI. No prior experience in ML and AI is needed. You should have some knowledge of linear algebra, introductory calculus and probability. Some programming experience is also recommended.

- [ ] [Machine Learning (Stanford CS229)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) 
  - [Course website](http://cs229.stanford.edu/syllabus-autumn2018.html)
  - This modern classic of machine learning courses is a great starting point to understand the concepts and techniques of machine learning. The course covers many widely used techniques, The lecture notes are detailed and review necessary mathematical concepts.
- [ ] [Convolutional Neural Networks for Visual Recognition (Stanford CS231n)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
  - [Course website](https://cs231n.github.io)
  - A great way to start with deep learning. The course focuses on convolutional neural networks and computer vision, but also gives an overview on recurrent networks and reinforcement learning.
- [ ] [Introduction to Artificial Intelligence (UC Berkeley CS188)](https://www.youtube.com/playlist?list=PL7k0r4t5c108AZRwfW-FhnkZ0sCKBChLH)
  - [Course website](https://inst.eecs.berkeley.edu/~cs188/fa18/index.html)
  - Covers the whole field of AI. From search methods, game trees and machine learning to Bayesian networks and reinforcement learning.
- [ ] [Applied Machine Learning 2020 (Columbia)](https://www.youtube.com/playlist?list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM)
  - Alternative to Stanford CS229. As the name implies, this course takes a more applied perspective than Andrew Ng's machine learning lecture at Stanford. You will see more code than mathematics. Concepts and algorithms are using the popular Python libraries scikit-learn and Keras.
- [ ] [Introduction to Reinforcement learning with David Silver (DeepMind)](https://www.youtube.com/playlist?list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB)
  - [UCL Course on Reinforcement Learning by David Silver ](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
  - [Course website](https://www.davidsilver.uk/teaching/)
  - Introduction to reinforcement learning by one of the leading researchers behind AlphaGo and AlphaZero.
- [ ] [Natural Language Processing with Deep Learning (Stanford CS224N)](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) 
  - [Course website](http://web.stanford.edu/class/cs224n/)
  - Modern NLP techniques from recurrent neural networks and word embeddings to transformers and self-attention. Covers applied topics like questions answering and text generation.

**Advanced Lectures:**

Advanced courses that require prior knowledge in machine learning and AI.

- [ ] [Deep Unsupervised Learning (UC Berkeley CS294)](https://www.youtube.com/channel/UCf4SX8kAZM_oGcZjMREsU9w/videos) 
  - [Course website](https://sites.google.com/view/berkeley-cs294-158-sp19/home)
- [ ] [Frontiers of Deep Learning (Simons Institute)](https://www.youtube.com/playlist?list=PLgKuh-lKre11ekU7g-Z_qsvjDD8cT-hi9)
  - [Course website](https://simons.berkeley.edu/workshops/dl2019-1)
- [ ] [New Deep Learning Techniques](https://www.youtube.com/playlist?list=PLHyI3Fbmv0SdM0zXj31HWjG9t9Q0v2xYN)
  - [Course website](http://www.ipam.ucla.edu/programs/workshops/new-deep-learning-techniques/?tab=overview)
- [ ] [Geometry of Deep Learning (Microsoft Research)](https://www.youtube.com/playlist?list=PLD7HFcN7LXRe30qq36It2XCljxc340O_d)
  - [Course website](https://www.microsoft.com/en-us/research/event/ai-institute-2019/)
- [ ] [Deep Multi-Task and Meta Learning (Stanford CS330)](https://www.youtube.com/playlist?list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5)
  - [Course Website](http://cs330.stanford.edu/)
- [ ] [Mathematics of Machine Learning Summer School 2019 (University of Washington)](https://www.youtube.com/playlist?list=PLTPQEx-31JXhguCush5J7OGnEORofoCW9)
  - [Course Website](http://mathofml.cs.washington.edu/)
- [ ] [Probabilistic Graphical Models (Carneggie Mellon University)](https://www.youtube.com/playlist?list=PLoZgVqqHOumTY2CAQHL45tQp6kmDnDcqn)
  - [Course Website](https://sailinglab.github.io/pgm-spring-2019/)
- [ ] [Probabilistic and Statistical Machine Learning 2020 (University of Tübingen)](https://www.youtube.com/playlist?list=PL05umP7R6ij1tHaOFY96m5uX3J21a6yNd)
- [ ] [Statistical Machine Learning 2020 (University of Tübingen)](https://www.youtube.com/playlist?list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC)
- [ ] [Mobile Sensing and Robotics 2019 (Bonn University)](https://www.youtube.com/playlist?list=PLgnQpQtFTOGQJXx-x0t23RmRbjp_yMb4v)
- [ ] [Sensors and State Estimation Course 2020 (Bonn University)](https://www.youtube.com/playlist?list=PLgnQpQtFTOGQh_J16IMwDlji18SWQ2PZ6)
- [ ] [Photogrammetry 2015 (Bonn University)](https://www.youtube.com/playlist?list=PLgnQpQtFTOGRsi5vzy9PiQpNWHjq-bKN1)
- [ ] [Advanced Deep Learning & Reinforcement Learning 2020 (DeepMind / UCL)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)
- [ ] [Data-Driven Dynamical Systems with Machine Learning](https://www.youtube.com/playlist?list=PLMrJAkhIeNNR6DzT17-MM1GHLkuYVjhyt)
- [ ] [Data-Driven Control with Machine Learning](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQkv98vuPjO2X2qJO_UPeWR)
- [ ] [ECE AI Seminar Series 2020 (NYU)](https://www.youtube.com/playlist?list=PLhwo5ntex8iY9xhpSwWas451NgVuqBE7U)
- [ ] [CS287 Advanced Robotics at UC Berkeley Fall 2019](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNBPJdt8WamRAt4XKc639wF)
- [ ] [CSEP 546 - Machine Learning (AU 2019) (U of Washington)](https://www.youtube.com/playlist?list=PLTPQEx-31JXj87XLsYutYGKw6K9dNaD36)
- [ ] [Deep Reinforcment Learning, Decision Making and Control (UC Berkeley CS285)](https://www.youtube.com/playlist?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A)
- [ ] [Stanford Convex Optimization](https://www.youtube.com/playlist?list=PLdrixi40lpQm5ksInXlRon1eRwq_gzIcw)
- [ ] [Stanford CS224U: Natural Language Understanding | Spring 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20)
- [ ] [Full Stack Deep Learning 2019](https://www.youtube.com/playlist?list=PL1T8fO7ArWlcf3Hc4VMEVBlH8HZm_NbeB)
- [ ] [Emerging Challenges in Deep Learning](https://www.youtube.com/playlist?list=PLgKuh-lKre10BpafDrv0fg2VNUweWXWVd)
- [ ] [Deep|Bayes 2019 Summer School](https://www.youtube.com/playlist?list=PLe5rNUydzV9QHe8VDStpU0o8Yp63OecdW)
- [ ] [CMU Neural Nets for NLP 2020](https://www.youtube.com/playlist?list=PL8PYTP1V4I8CJ7nMxMC8aXv8WqKYwj-aJ)
- [ ] [New Directions in Reinforcement Learning and Control (Institure for Advanced Study)](https://www.youtube.com/playlist?list=PLdDZb3TwJPZ61sGqd6cbWCmTc275NrKu3)
- [ ] [Workshop on Theory of Deep Learning: Where next (Institure for Advanced Study)](https://www.youtube.com/playlist?list=PLdDZb3TwJPZ5dqqg_S-rgJqSFeH4DQqFQ)
- [ ] [Deep Learning: Alchemy or Science? (Institure for Advanced Study)](https://www.youtube.com/playlist?list=PLdDZb3TwJPZ7aAxhIHALBoh8l6-UxmMNP)
- [ ] [Theoretical Machine Learning Lecture Series (Institure for Advanced Study)](https://www.youtube.com/playlist?list=PLdDZb3TwJPZ5VLprf2VUfC0h1zOGvV_gz)

## 🔹 Coursera Courses

- [ ] [Spesialization: Advanced Machine Learning ](https://www.coursera.org/specializations/aml)


## 🟥 YouTube Videos

- [ ] [TensorFlow: Coding TensorFlow](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwLbzbnKJ59NkZvQAW9wLbx) playlist
- [ ] [Google Cloud Platform: AI Adventures](https://www.youtube.com/playlist?list=PLIivdWyY5sqJxnwJhe3etaK7utrBiPBQ2)
- [ ] [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw/playlists) channel

## 📚 Books

- [ ] [An Introduction to Statistical Learning](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/Tensor%20Calculus)
- [ ] [Tensor Calculus](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf)
- [ ] [The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition (Springer Series in Statistics)](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576/ref=asc_df_0387848576/?tag=hyprod-20&linkCode=df0&hvadid=312140868236&hvpos=1o2&hvnetw=g&hvrand=14103077512698652064&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9031958&hvtargid=aud-801738734305:pla-487139763557&psc=1)
 - [ ] [Pattern Recognition and Machine Learning, Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

## ▶️ Websites

- [ ] [Talking Machines](https://www.thetalkingmachines.com)Talking Machines

## 📁 GitHub Repositories

- [ ] [100-Days-Of-ML-Code](https://github.com/Avik-Jain/100-Days-Of-ML-Code)
    

## 📌 Other
- [ ] [Tucker Decomposition](https://en.wikipedia.org/wiki/Tucker_decomposition), [Article](https://link.springer.com/article/10.1007/BF02289464)
 - [ ] [Katakoda](https://www.katacoda.com/)
 - [ ] [A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
    - [Code](https://drive.google.com/file/d/16hiZefHVkIb8Cvhf3BbBXsOOV9hob7eo/view)

    
## Reinforcement Learning 

- [ ] Books:
   - [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)
- [ ] Classes:
   - [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
      - [Introduction to Reinforcement learning with David Silver (DeepMind)](https://www.youtube.com/playlist?list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB)
      - [Course website](https://www.davidsilver.uk/teaching/)
      - Introduction to reinforcement learning by one of the leading researchers behind AlphaGo and AlphaZero.
   - [CS294 - Deep Reinforcement Learning (Berkeley, Fall 2015)](http://rail.eecs.berkeley.edu/deeprlcourse/)
   - [CS 8803 - Reinforcement Learning (Georgia Tech)](https://www.udacity.com/course/reinforcement-learning--ud600)
   - [CS885 - Reinforcement Learning (UWaterloo), Spring 2018](https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/)
   - [CS294-112 - Deep Reinforcement Learning (UC Berkeley)](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [ ] Task / Tutorials:
   - [Introduction to Reinforcement Learning (Joelle Pineau @ Deep Learning Summer School 2016)](http://videolectures.net/deeplearning2016_pineau_reinforcement_learning/)
   - [Deep Reinforcement Learning (Pieter Abbeel @ Deep Learning Summer School 2016)](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/)
   - [Deep Reinforcement Learning ICML 2016 Tutorial (David Silver)](http://techtalks.tv/talks/deep-reinforcement-learning/62360/)
   - [Tutorial: Introduction to Reinforcement Learning with Function Approximation](https://www.youtube.com/watch?v=ggqnxyjaKe4)
   - [John Schulman - Deep Reinforcement Learning (4 Lectures)](https://www.youtube.com/playlist?list=PLjKEIQlKCTZYN3CYBlj8r58SbNorobqcp)
   - [Deep Reinforcement Learning Slides @ NIPS 2016](http://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf)
   - [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)
   - [Advanced Deep Learning & Reinforcement Learning (UCL 2018, DeepMind) -Deep RL Bootcamp](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)
- [ ] GitHub Repositories:
   - [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning), Implementation of Reinforcement Learning Algorithms. Python, OpenAI Gym, Tensorflow. Exercises and Solutions to accompany Sutton's Book and David Silver's course.
- [ ] [Deep Reinforcment Learning, Decision Making and Control (UC Berkeley CS285)](https://www.youtube.com/playlist?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A)
- [ ] [New Directions in Reinforcement Learning and Control (Institure for Advanced Study)](https://www.youtube.com/playlist?list=PLdDZb3TwJPZ61sGqd6cbWCmTc275NrKu3)


## 📑 Datasets list

The initial list was provided by Kevyn Collins-Thomson from the University of Michigan School of Information.

- **Long general-purpose list of datasets:** 
    - https://vincentarelbundock.github.io/Rdatasets/datasets.html

- This website has dozens of public datasets - some fun, some a bit, well.. quirky. external link: 
    - https://rs.io/100-interesting-data-sets-for-statistics/

- **The Academic Torrents site** has a growing number of datasets, including a few text collections that might be of interest (Wikipedia, email, twitter, academic, etc.) for current or future projects. 
    - http://academictorrents.com/browse.php?cat=6

- **Google Books n-gram corpus**
    - External link: http://books.google.com/ngrams
    - Dataset: external link: http://aws.amazon.com/datasets/8172056142375670
    
- **Common Crawl:** • Currently 6 billion Web documents (81 Tb) • Amazon S3 Public Data Set
    - http://aws.amazon.com/datasets/41740
    - https://commoncrawl.atlassian.net/wiki/display/CRWL/About+the+Data+Set
    - Award project using Common Crawl: http://norvigaward.github.io/entries.html
    - Python example: http://www.freelancer.com/projects/Python-Data-Processing/Python-script-for-CommonCrawl.html


- **Business/commercial data Yelp external link:**
    - http://www.yelp.com/developers/documentation/v2/search_api
    - Upcoming Deprecation of Yelp API v2 on June 30, 2018 (Posted by Yelp Jun 28, 2017)
    
- **Internet Archive** (huge, ever-growing archive of the Web going back to 1990s) external link:
    - http://archive.org/help/json.php
    
- **WikiData:**
    - https://www.wikidata.org/wiki/Wikidata:Main_Page

- **World Food Facts**
    - http://world.openfoodfacts.org/data

- **Data USA - a variety of census data**
    - https://datausa.io/
    
- **U.S. Government open data** - datasets from 75 agencies and subagencies
    - https://data.gov/
    
- **NASA data portal** - space and earth science
    - https://data.nasa.gov/ 
 
## What's is the difference between _train, validation and test set_, in neural networks?

**Training Set:** this data set is used to adjust the weights on the neural network.

**Validation Set:** this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over the validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.

> The **validation data set** is a set of data for the function you want to learn, which you are not directly using to train the network. You are training the network with a set of data which you call the training data set. If you are using gradient based algorithm to train the network then the error surface and the gradient at some point will completely depend on the training data set thus the training data set is being directly used to adjust the weights. To make sure you don't overfit the network you need to input the validation dataset to the network and check if the error is within some range. Because the validation set is not being using directly to adjust the weights of the network, therefore a good error for the validation and also the test set indicates that the network predicts well for the train set examples, also it is expected to perform well when new example are presented to the network which was not used in the training process.

**Testing Set:** this data set is used only for testing the final solution in order to confirm the actual predictive power of the network.

**Also**, in the case you do not have enough data for a validation set, you can use **cross-validation** to tune the parameters as well as estimate the test error.

**Cross-validation set** is used for model selection, for example, select the polynomial model with the least amount of errors for a given parameter set. The test set is then used to report the generalization error on the selected model. 

**[Early stopping](https://en.wikipedia.org/wiki/Early_stopping)** is a way to stop training. There are different variations available, the main outline is, both the train and the validation set errors are monitored, the train error decreases at each iteration ([backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and brothers) and at first the validation error decreases. The training is stopped at the moment the validation error starts to rise. The weight configuration at this point indicates a model, which predicts the training data well, as well as the data which is not seen by the network . But because the validation data actually affects the weight configuration indirectly to select the weight configuration. This is where the Test set comes in. This set of data is never used in the training process. Once a model is selected based on the validation set, the test set data is applied on the network model and the error for this set is found. This error is a representative of the error which we can expect from absolutely new data for the same problem.
 
# ⚙️ **Models and Algorithms Implementation:**

- [Models and Algorithms Implementation](https://github.com/ElizaLo/Machine-Learning#%EF%B8%8F-models-and-algorithms)
  1. [k Nearest Neighbor](https://github.com/ElizaLo/ML-with-Jupiter#k-nearest-neighbor)
  2. [Linear Regression](https://github.com/ElizaLo/ML-with-Jupiter#linear-regression)
      - [Polynomial Regression](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#polynomial-regression)
  3. [Logistic Regression](https://github.com/ElizaLo/ML-with-Jupiter#logistic-regression)
  4. [Fully Connected Neural Networks](https://github.com/ElizaLo/ML-with-Jupiter#fully-connected-neural-networks)
  5. [Computer Vision - Blurred Images](https://github.com/ElizaLo/ML-with-Jupiter#computer-vision---blurred-images)
  6. [OpenCV](https://github.com/ElizaLo/ML-with-Jupiter#opencv)
  7. [Convolutional Neural Network (CNN)](https://github.com/ElizaLo/ML-with-Jupiter#convolutional-neural-network-cnn)
  8. [Gated Recurrent Units (GRU)](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab#gated-recurrent-units-gru)

 1. ## **k Nearest Neighbor**
    - [Code](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/k%20Nearest%20Neighbor/kNN.py)
    - [Subset of MNIST](https://pjreddie.com/projects/mnist-in-csv/)

 2. ## **Linear Regression**
    - 📘 [Math](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#linear-regression)
      - ["Batch" Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#batch-gradient-descent)
      - [Gradient Descent For Linear Regression](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#gradient-descent-for-linear-regression)
      - [Gradient Descent For Multiple Variables](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#gradient-descent-for-multiple-variables)
    - 📘 [Polynomial Regression](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#polynomial-regression)
    - 💻 [Code](https://github.com/ElizaLo/ML-with-Jupiter/tree/master/P2)
    
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
    
 8. ## **Gated Recurrent Units (GRU)**
    - [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be), Towards Data Science
 
 
# 👩‍💻 Projects: 

 - # **Spam Detection**
 
   ⚙️ Methods:
    - **_Naive Bayes spam filtering_**
    - **_K-Nearest Neighbors algorithm_**
    - **_Decision Tree learning_**
    - **_Support Vector Machine (SVM)_**
    - **_Random Forest_**
    
 - 💻 [Code](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Spam%20Detection/Spam_Detection.ipynb)
 - [SMS Spam Collection Dataset](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Spam%20Detection/spam.csv)
  
 - # **Text Generator**
   
   Neural Network for generating text based on training txt file using **_Google Colab_**. 
   As a base text were used **_Alice in Wonderland_** by Lewis Carroll.
   
    - 💻 [Code](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/Text%20Generator/Text_Generator.ipynb)
    - [Base text - **Alice in Wonderland**](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Text%20Generator/alice_in_wonderland.txt)
    - [Formatted text of **Alice in Wonderland**](https://github.com/ElizaLo/ML-with-Jupiter/blob/master/Text%20Generator/alice_formatted.txt)
    
 - # Question Answering System using BiDAF Model on SQuAD
   
   Implemented a Bidirectional Attention Flow neural network as a baseline on SQuAD, improving Chris Chute's model [implementation](https://github.com/chrischute/squad/blob/master/layers.py), adding word-character inputs as described in the original paper and improving [GauthierDmns' code](https://github.com/GauthierDmn/question_answering).
   
    - [Project Repository](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD)
    - [Paper](https://github.com/ElizaLo/NLP/blob/master/Question%20Answering%20System/Question%20Answering%20System%20based%20on%20SQuAD.pdf)
    - [Used Articles](https://github.com/ElizaLo/NLP/tree/master/Question%20Answering%20System/Articles)
    - [Useful Articles](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD#useful-articles)
    - [Useful Links](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD#useful-links)
  
