import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from katas.linear import LinearRegression


def test__linear_regression__predictions():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = X @ np.array([1, 2]) + 3

    linear_regression: LinearRegression = LinearRegression()

    linear_regression.fit(X, y)

    #print(linear_regression.coefficients)
    #print(linear_regression.predict(X))
    assert_array_equal(linear_regression.predict(np.array([[3, 5]])), np.array([[16.40578669]]))
    #print(linear_regression.score(X, y))
