# Linear Regression

## Cost Function

We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from _x_'s and the actual output _y_'s.

Function that shows how accurate the predictions of the hypothesis are with current set of parameters.

![Cost Function](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/cost_function.png)

_x<sup>i</sup>_ - input (features) of _i<sup>th</sup>_ training example

_y<sup>i</sup>_ - output of _i<sup>th</sup>_ training example

_m_ - number of training examples

![difference](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/difference.png) - difference between the predicted value and the actual value

This function is otherwise called the **"Squared error function"**, or **"Mean squared error"**. The mean is halved 1/2 as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the 1/2 term. 

## "Batch" Gradient Descent

Gradient descent is an iterative optimization algorithm for finding the minimum of a cost function described above. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.

> **"Batch"**: Each step of gradient descent uses **all** the traning examples.

![Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%201.png)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate.

For example, the distance between each _'star'_ in the graph above represents a step determined by our parameter _**α**_. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of ![Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/J(theta_0%2C%20theta_1).png). Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

**The gradient descent algorithm is:**

repeat until convergence:

![Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%20Formula.png)

where _j=0,1_ represents the feature index number.

At each iteration _j_, one should simultaneously update the parameters theta. Updating a specific parameter prior to calculating another one on the _j<sup>(th)</sup>_ iteration would yield to a wrong implementation.

- The gradient descent can converge to a local minimum, even with the learning rate  _**α fixed**_.
- As we approach a local minimum, gradient descent will automatically take smaller steps. So no need to decrease α over time.

![Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%202.png)

![Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%204.gif)

- If _**α is too large**_, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.
- If _**α is too small**_, gradient descent can be slow.

![Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%203.png)

![Gradient Descent](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%205.gif)

## Gradient Descent For Linear Regression

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :

![Gradient Descent for Linear Regression](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%20for%20Linear%20Regression.png)

where _**m**_ is the size of the training set, _**theta<sub>0</sub>**_ a constant that will be changing simultaneously with _**theta<sub>1</sub>**_ and  _**x<sub>i</sub>**_, _**y<sub>i</sub>**_ are values of the given training set (data).

- Note that we have separated out the two cases for  _**theta<sub>j</sub>**_ into separate equations for _**theta<sub>0</sub>**_ and _**theta<sub>1</sub>**_.

![Formula 1](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Formula%201.png)

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

## Gradient Descent For Multiple Variables

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

![Gradient Descent for Linear Regression](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%20for%20Multiple%20Variables%201.png)

In other words:

![Gradient Descent for Linear Regression](https://github.com/ElizaLo/ML-using-Jupiter-Notebook-and-Google-Colab/blob/master/P2/images/Gradient%20Descent%20for%20Multiple%20Variables%202.png)
