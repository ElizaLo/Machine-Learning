# Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

In supervised learning we have a set of training data as an input and a set of labels or "correct answers" for each training set as an output. Then we're training our model (machine learning algorithm parameters) to map the input to the output correctly (to do correct prediction). The ultimate purpose is to find such model parameters that will successfully continue correct input→output mapping (predictions) even for new input examples.

Supervised learning problems are categorized into **"regression"** and **"classification"** problems. 
- [ ] In a **regression problem**, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. 
- [ ] In a **classification problem**, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

_**Example 1:**_

- Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

_**Example 2:**_

- **Regression** - Given a picture of a person, we have to predict their age on the basis of the given picture

- **Classification** - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

### ▶️ Regression

In regression problems we do real value predictions. Basically we try to draw a line/plane/n-dimensional plane along the training examples. 

💻 _**Usage examples:** stock price forecast, sales analysis, dependency of any number, etc._

🔵 [**_Linear Regression_**](https://github.com/ElizaLo/ML-with-Jupiter#linear-regression)
  - 📘 [Math](https://github.com/ElizaLo/Machine-Learning-Course-Octave/tree/master/Linear%20Regression#linear-regression) - theory and [useful links](https://github.com/ElizaLo/Machine-Learning-Course-Octave/tree/master/Linear%20Regression#-references)
  
🔵 [**_Polynomial Regression_**](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/tree/master/P2#polynomial-regression)
  
### ▶️ Classification

In classification problems we split input examples by certain characteristic.

💻 _**Usage examples:** spam-filters, language detection, finding similar documents, handwritten letters recognition, etc._
