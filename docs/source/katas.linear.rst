Linear models
#############

The linear models module contains several popular instances of the generalized linear model (GLM).

Linear Regression
_________________

- :class:`~katas.linear.LinearRegression`

The simple linear regression model is

.. math::

    \mathbf{y} = \mathbf{bX} + \mathbf{\epsilon}

where:

.. math::

    \epsilon \sim \mathcal{N}(0, \sigma^2 I)

In probabilistic terms this corresponds to

.. math::

    \mathbf{y} - \mathbf{bX}  &\sim  \mathcal{N}(0, \sigma^2 I) \\
    \mathbf{y} \mid \mathbf{X}, \mathbf{b}  &\sim  \mathcal{N}(\mathbf{bX}, \sigma^2 I)

The loss for the model is simply the squared error between the model
predictions and the true values:

.. math::

    \mathcal{L} = ||\mathbf{y} - \mathbf{bX}||_2^2

The MLE for the model parameters **b** can be computed in closed form via
the normal equation:

.. math::

    \mathbf{b}_{MLE} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}

where :math:`(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top` is known
as the pseudoinverse / Moore-Penrose inverse.

Logistic Regression
___________________

- :class:`~katas.linear.LogisticRegression`

A simple logistic regression model fit via gradient descent on the
penalized negative log likelihood.

For logistic regression, the penalized negative log likelihood of the
targets **y** under the current model is

.. math::
    - \log \mathcal{L}(\mathbf{b}, \mathbf{y}) = -\frac{1}{N} \left[
        \left(
            \sum_{i=0}^N y_i \log(\hat{y}_i) +
              (1-y_i) \log(1-\hat{y}_i)
        \right) - R(\mathbf{b}, \gamma)
    \right]

where:

.. math::
    R(\mathbf{b}, \gamma) = \left\{
        \begin{array}{lr}
            \frac{\gamma}{2} ||\mathbf{beta}||_2^2 & :\texttt{ penalty = 'l2'}\\
            \gamma ||\beta||_1 & :\texttt{ penalty = 'l1'}
        \end{array}
        \right.

is a regularization penalty, :math:`\gamma` is a regularization weight,
`N` is the number of examples in **y**, and **b** is the vector of model
coefficients.

.. toctree::
   :maxdepth: 2
   :hidden:

   katas.linear.index
