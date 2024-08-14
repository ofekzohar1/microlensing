import math
import numpy as np
import numpy.typing as npt
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import norm

NDAfloat = npt.NDArray[np.float_]

########################################### Classes ###########################################

class value_with_error:
    """The class represents a measured value and its error

    Attributes:
        name (`str`): The value's name
        value (`float`): The numerical value
        error (`float`): The numerical error
    """
    def __init__(self,name: str, value: float, error: float) -> None:
        self.name = name
        self.value = float(value)
        self.error = float(error)

    def __str__(self) -> str:
        # Calculate fixed point precision - 2 most significant digits of the error
        order = orderOfMagnitude(self.error)
        precision = max(np.abs(order) + 1, 3)  # No less than 3 digits...

        # Return in Physics lab representation
        return f"{self.name}: {self.value:.{precision}f}\u00B1{self.error:.{precision}f}"

    def __repr__(self) -> str:
        return str(self)


########################################### Helper Functions ###########################################
   
def independent_meas_linear_fit(n_param: int, x: npt.ArrayLike, y: npt.ArrayLike, y_error: npt.ArrayLike) -> Tuple[NDAfloat, NDAfloat, NDAfloat, float]:
    """Apply linear least sq fit to the given data

    Args:
        n_param (`int`): The number of fitted params
        x (`ArrayLike`): x measures
        y (`ArrayLike`): y measures
        y_error (`ArrayLike`): y errors

    Returns:
        NDAfloat: n_param len 1-D array contains the parameters minimizing chi
        NDAfloat: n_param len 1-D array contains the parameters' errors
        NDAfloat: lne(y) 1-D array contains the estimated y using the best params
        float: the chi value
    """
    # Build the param coefficients matrix - f=Ca
    C = np.ndarray((len(x), 0))
    for i in range(n_param):
        C = np.column_stack((C, x ** i))

    var_y_inv = np.diag(1 / (y_error ** 2))         # V^-1 - the inv var matrix of y
    inter_res = C.T @ var_y_inv                     # intermediate result - C^T * V^-1
    var_param_est = np.linalg.inv(inter_res @ C)    # The params var matrix - (C^T * V^-1 * C)^-1
    pararm_est = var_param_est @ inter_res @ y      # The params vector - (C^T * V^-1 * C)^-1 * C^T * V^-1 * y
    y_est = C @ pararm_est                          # The calculated y=Ca

    # calc chi sq red
    residue_mat = y - y_est
    ddof = max(len(x) - n_param, 1)
    chi_sq_red = residue_mat.T @ var_y_inv @ residue_mat / ddof
    
    return pararm_est, np.diag(var_param_est) ** 0.5, y_est, chi_sq_red

def orderOfMagnitude(num: float) -> int:
    """Return the order of the given number"""
    if num == 0:
        return 0
    return math.floor(math.log(num, 10))

def sqrt_sum_of_sq(a: npt.ArrayLike) -> float:
    """Calculate the square root of the sum of squares of the array elements

    Args:
        a (`ArrayLike`): The elements to be squared

    Returns:
        float: square root of the sum of squares
    """
    return np.sqrt(np.sum(np.array(a) ** 2))

def nsigma(expected: value_with_error, meas: value_with_error) -> float:
    """Calculate the n-sigma test between measured and expected values

    Args:
        expected (`value_with_error`): The expected value and its error
        meas (`value_with_error`): The measured value and its error

    Returns:
        float: n-sigma test
    """
    return np.abs(expected.value-meas.value) / sqrt_sum_of_sq([expected.error, meas.error])

def error_combination(derivative: npt.ArrayLike, error: npt.ArrayLike) -> float:
    """Calculate the error combination of ind. errors

    Args:
        derivative (`ArrayLike`): The calculated partial derivative W.R.T to the errors (weights)
        error (`ArrayLike`): The errors to be combined

    Returns:
        float: The error combination
    """
    return sqrt_sum_of_sq(np.multiply(derivative, error))

def norm_hist(name: str, data: npt.ArrayLike) -> value_with_error:
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the histogram
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g', ec='black')
    plt.xlabel(name)

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.4f,  std = %.4f" % (mu, std)
    plt.title(title)
    plt.show()

    return value_with_error(name+"_hist", mu, std)

def bootstrap_compare(fit: value_with_error, hist: value_with_error) -> float:
    res = np.abs(fit.value-hist.value) / fit.error
    
    print(fit)
    print(hist)
    print(f"bootstrap value comparison: {res}")
    print(f"bootstrap error comparison: fit error order e{orderOfMagnitude(fit.error)}, bootstrap error order e{orderOfMagnitude(hist.error)}")

    return res

def residue_plot(xlabel: str, ylabel: str, x: npt.ArrayLike, y: npt.ArrayLike, y_error: npt.ArrayLike, y_est: npt.ArrayLike):
    plt.grid()
    plt.errorbar(x=x, y=y-y_est, yerr=y_error, fmt='o', markersize=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y = 0, linestyle = '--')
    plt.show()