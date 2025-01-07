import math
import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import norm

NDAfloat = npt.NDArray[np.float_]
TargetFunc = Callable[[List[float], npt.ArrayLike], npt.ArrayLike]

#######################################################################################################
############################################### Classes ###############################################
#######################################################################################################

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

class Param:
    def __init__(self, name: str, min: float, max: float, n: int = 17) -> None: #n should be odd
        self.name = name
        self.min = min
        self.max = max
        self.n = n
    
    def range(self):
        return np.linspace(self.min,self.max,num=self.n)
    
    def get_new_range(self, value):
        width = (self.max-self.min) / (self.n-1)
        #print (self, " | " ,value, " | ",  Param(max(value - width, self.min), min(value+width, self.max)))
        return Param(self.name, max(value - width, self.min), min(value+width, self.max))
        #return Param(value - width, value+width)
    
    def __str__(self):
        return f"param {self.name}: min={self.min}, max={self.max}"

class MeshgridChiMinNonLinearFit:
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, y_err: npt.ArrayLike, target_func: TargetFunc):
        self.x = x
        self.y = y
        self.y_err = y_err
        self.target_func = target_func

    def fit(self, init_params: List[Param], fixed_params: List[Param], res_chi: float = 0.00001):
        min_chi = float('inf')
        counter = 0
        new_min_chi = 1000000000000000000000000
        
        all_chis = []
        all_params_comb = []
        curr_params = init_params
        while (min_chi - new_min_chi) > res_chi:
            min_chi = new_min_chi

            curr_params,  chis , params_combinations = self._min_chi_on_params(curr_params, fixed_params)
            all_params_comb.extend(params_combinations)
            all_chis.extend(chis)
            new_min_chi = min(chis)
            min_param_comb = params_combinations[np.argmin(chis)]
            thepupik = " ".join(str(s) for s in init_params)
            #print(f"{counter} \n old chi: {min_chi} | new chi: {n_min_chi} | min_comb: {min_param_comb}\n param: {thepupik} \n")
            
            counter +=1

        #print(f"finished!! chi: {n_min_chi} | min_comb: {min_param_comb} \n")
        return all_chis, all_params_comb

    def _min_chi_on_params(self, params: List[Param], fixed_params: List[Param]):
        new_params = []  # list of new parameters range

        fixed_params_dict = {param.name: param.min for param in fixed_params}

        params_combinations = np.array(np.meshgrid(*[param.range() for param in params])).reshape(-1,len(params))
        chis = [calc_chi_sq(self.x, self.y, self.y_err, {params[i].name: param for i, param in enumerate(comb)} | fixed_params_dict, self.target_func) for comb in params_combinations]

        min_param_comb = params_combinations[np.argmin(chis)]
        for i in range(len(min_param_comb)):
            new_params.append(params[i].get_new_range(min_param_comb[i]))

        return new_params, chis , params_combinations

#######################################################################################################
########################################## Utility Functions ##########################################
#######################################################################################################

########################################## Fit Functions ##########################################

def independent_meas_linear_fit(n_param: int, x: npt.ArrayLike, y: npt.ArrayLike, y_err: npt.ArrayLike) -> Tuple[NDAfloat, NDAfloat, NDAfloat, float]:
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

    var_y_inv = np.diag(1 / (y_err ** 2))         # V^-1 - the inv var matrix of y
    inter_res = C.T @ var_y_inv                     # intermediate result - C^T * V^-1
    var_param_est = np.linalg.inv(inter_res @ C)    # The params var matrix - (C^T * V^-1 * C)^-1
    pararm_est = var_param_est @ inter_res @ y      # The params vector - (C^T * V^-1 * C)^-1 * C^T * V^-1 * y
    y_est = C @ pararm_est                          # The calculated y=Ca

    # calc chi sq red
    residue_mat = y - y_est
    ddof = max(len(x) - n_param, 1)
    chi_sq_red = residue_mat.T @ var_y_inv @ residue_mat / ddof
    
    return pararm_est, np.diag(var_param_est) ** 0.5, y_est, chi_sq_red

def do_func(params: Dict[str, float], x: npt.ArrayLike) -> npt.ArrayLike:
    u_min = params["u_min"]
    t0 = params["t0"]
    tau = params["tau"]
    f_bl = params["f_bl"]
    u_t = np.sqrt(u_min ** 2 + ((x-t0)/tau) ** 2)
    mu = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    return f_bl * (mu - 1) + 1

def calc_chi_sq(x: npt.ArrayLike, y: npt.ArrayLike, y_error: npt.ArrayLike, params: Dict[str, float], target_func: TargetFunc) -> float:
    return sum_of_sq((target_func(params, x) - y) / y_error)

def orderOfMagnitude(num: float) -> int:
    """Return the order of the given number"""
    if num == 0:
        return 0
    return math.floor(math.log(num, 10))

def sum_of_sq(a: npt.ArrayLike) -> float:
    """Calculate the sum of squares of the array elements

    Args:
        a (`ArrayLike`): The elements to be squared

    Returns:
        float: the sum of squares
    """
    return np.sum(np.array(a) ** 2)

def sqrt_sum_of_sq(a: npt.ArrayLike) -> float:
    """Calculate the square root of the sum of squares of the array elements

    Args:
        a (`ArrayLike`): The elements to be squared

    Returns:
        float: square root of the sum of squares
    """
    return np.sqrt(sum_of_sq(a))

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

def mu_from_I_and_fbl(I: value_with_error, fbl: value_with_error) -> value_with_error:
        mu_value = (I.value-1) / fbl.value + 1
        mu_derivative_wrt_I = 1 / fbl.value
        mu_derivative_wrt_fbl = (I.value-1) / (fbl.value ** 2)
        mu_error = error_combination([mu_derivative_wrt_I, mu_derivative_wrt_fbl], [I.error, fbl.error])

        return value_with_error("mu", mu_value, mu_error)

def u_min_from_mu_max(mu: value_with_error) -> value_with_error:
        u_value = math.sqrt(2 * (mu.value / math.sqrt(mu.value**2 - 1) - 1))
        
        u_derivative_wrt_mu = 1 / (u_value * ((mu.value**2 - 1) ** 1.5))
        u_error = error_combination([u_derivative_wrt_mu], [mu.error])

        return value_with_error("umin", u_value, u_error)