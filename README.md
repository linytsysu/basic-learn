# basic-learn

basic-learn is a Python module which includes basic implementation of machine learning algorithms. basic-learn is built on top of [NumPy](http://www.numpy.org/), which is a fundamental package for scientific computing with Python.

### Contents

+ <a href="#linear_regression">linear regression</a>


#### [<span id="linear_regression">Linear Regression</a>](https://linytsysu.github.io/2018/06/03/ml-linear-regression.html)

+ formula:

    <img src="https://latex.codecogs.com/svg.latex?%5Ctheta_i%20%3D%20%5Ctheta_i%20-%20%5Calpha%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bj%3D1%7D%5E%7Bm%7D%28h_%5Ctheta%28%5Cboldsymbol%7Bx%7D_j%29%20-%20y_j%29x_i%5E%7Bj%7D" title="\theta_i = \theta_i - \alpha\frac{1}{m}\sum\limits_{j=1}^{m}(h_\theta(\boldsymbol{x}_j) - y_j)x_i^{j}" />

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
