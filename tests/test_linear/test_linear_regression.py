import numpy as np
from numpy.testing import assert_array_almost_equal

from katas.linear import LinearRegression


def test__linear_regression__predictions():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = X @ np.array([1, 2]) + 3

    linear_regression = LinearRegression()

    linear_regression.fit(X, y)

    assert_array_almost_equal(
        linear_regression.params,
        np.array([[2.51833456], [1.30186925], [1.99636888]]),
    )
    assert_array_almost_equal(
        linear_regression.predict(np.array([[3, 5]])), np.array([[16.40578669]])
    )
