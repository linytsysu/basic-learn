# basic-learn

basic-learn is a Python module which includes basic implementation of machine learning algorithms. basic-learn is built on top of [NumPy](http://www.numpy.org/), which is a fundamental package for scientific computing with Python.

### Contents

+ <a href="#linear_regression">linear regression</a>


#### [<span id="linear_regression">Linear Regression</a>](https://linytsysu.github.io/2018/06/03/ml-linear-regression.html)

+ formula:

    <img src="https://latex.codecogs.com/svg.latex?\theta_i&space;=&space;\theta_i&space;-&space;\alpha\frac{1}{m}\sum\limits_{j=0}^{m}(h_\theta(x_0^{(j)},&space;x_1^{(j)},&space;\ldots,&space;x_n^{(j)})&space;-&space;y_j)x_i^{(j)}" title="\theta_i = \theta_i - \alpha\frac{1}{m}\sum\limits_{j=0}^{m}(h_\theta(x_0^{(j)}, x_1^{(j)}, \ldots, x_n^{(j)}) - y_j)x_i^{(j)}" />

+ code:

    ``` python
    class LinearRegression:
        def __init__(self):
            pass

        def fit(self, X, y, alpha=0.01, loop=1000):
            m, n = X.shape
            self.theta = np.zeros(n)
            for _ in range(loop):
                hypothesis = np.dot(X, self.theta)
                loss = hypothesis - y
                gradient = np.dot(X.T, loss)
                self.theta = self.theta - alpha / m * gradient
            return self

        def predict(self, X):
            return np.dot(X, self.theta)
    ```
