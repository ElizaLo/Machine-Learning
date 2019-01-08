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
