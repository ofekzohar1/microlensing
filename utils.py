import math
import time
import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from scipy.stats import norm
from consts import *
import datetime

############################################### Types ###############################################

NDAfloat = npt.NDArray[np.float_]
FloatDict = Dict[str, float]
TargetFunc = Callable[[FloatDict, npt.ArrayLike], npt.ArrayLike]

#######################################################################################################
############################################### Classes ###############################################
#######################################################################################################

class ValueWithError:
    """The class represents a measured value and its error

    Attributes:
        name (`str`): The value's name
        value (`float`): The numerical value
        error (`float`): The numerical error
    """
    def __init__(self,name: str, value: float, upper_error: float, lower_error: float = None) -> None:
        self.name = name
        self.value = value
        self.upper_error = upper_error
        self.lower_error = lower_error if lower_error is not None else upper_error

    def __str__(self) -> str:
        # Calculate fixed point precision - 2 most significant digits of the error
        precision = self._calc_precision()

        # Return in Physics lab representation
        s = f"{self.name}: {self.str_value()}"
        if f"{self.upper_error:.{precision}f}" == f"{self.lower_error:.{precision}f}":
            s += f"\u00B1{self.upper_error:.{precision}f}"
        else:
            s += f"[-{self.lower_error:.{precision}f},+{self.upper_error:.{precision}f}]"
        return s

    def str_value(self) -> str:
        return f"{self.value:.{self._calc_precision()}f}"

    def _calc_precision(self) -> int:
        # Calculate fixed point precision - 2 most significant digits of the error
        order = max(orderOfMagnitude(self.upper_error), orderOfMagnitude(self.lower_error))
        return max(np.abs(order) + 1, 2)  # No less than 2 digits...
        

    def __repr__(self) -> str:
        return str(self)

ValErrDict = Dict[str, ValueWithError]
class ParamRange:
    def __init__(self, name: str, min: float, max: float, n_segments: int = 10, width_res: float = 0) -> None: #n should be odd
        self.name = name
        self.min = min
        self.max = max
        self.n_segments = n_segments
        self.width_res = width_res
    
    def range(self):
        return np.linspace(self.min,self.max,num=self.n_segments+1)
    
    def get_new_range(self, value):
        width = (self.max-self.min) / self.n_segments
        if width < self.width_res:
            width = 0
        #print (self, " | " ,value, " | ",  Param(max(value - width, self.min), min(value+width, self.max)))
        return ParamRange(self.name, max(value - width, self.min), min(value+width, self.max), self.n_segments, self.width_res)
        #return Param(value - width, value+width)
    
    def __str__(self):
        return f"param {self.name}: min={self.min}, max={self.max}, n_seg={self.n_segments}, width_res={self.width_res}"

class MeshgridChiMinNonLinearFit:
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, y_err: npt.ArrayLike, target_func: TargetFunc):
        self.x = x
        self.y = y
        self.y_err = y_err
        self.target_func = target_func
        self.df_params_comb: pd.DataFrame = None
        self.fit_params: ValErrDict = {}
        self.fit_chi = -1.0

    def fit(self, init_params: List[ParamRange], fixed_params: FloatDict, res_chi: float = 0, max_iters: int = np.inf) -> Tuple[ValErrDict, float]:
        fit_params, self.fit_chi, self.df_params_comb = self._meshgrid_fit(init_params=init_params, fixed_params=fixed_params, res_chi=res_chi, max_iters=max_iters)
        self.fit_params = self._fit_params_errors2(self.df_params_comb, fit_params)


        return self.fit_params, self.fit_chi

    def _meshgrid_fit(self, init_params: List[ParamRange], fixed_params: FloatDict, res_chi: float = 0, max_iters: int = np.inf) -> Tuple[FloatDict, float, pd.DataFrame]:
        min_chi, new_min_chi = float('inf'), float('inf')
        counter = 0
        
        iter_df_list = []
        col_names = [param.name for param in init_params]
        
        curr_params_range = init_params
        min_params_comb = []
        while (min_chi - new_min_chi) > res_chi or min_chi == float('inf'):
            #thepupik = "\n".join(str(s) for s in curr_params_range)
            #print(f"{counter} \n old chi: {min_chi} | new chi: {new_min_chi} | min_comb: {min_params_comb}\n param: {thepupik}")
            
            min_chi = new_min_chi
            min_params = {init_params[i].name: min_param for i, min_param in enumerate(min_params_comb)}

            if counter >= max_iters:
                break

            chis, params_combinations = self._calc_chi_on_params_meshgrid(curr_params_range, fixed_params)

            # Accumulate all chis and params combinations from all iterations
            iter_df = pd.DataFrame(data=params_combinations, columns=col_names)
            iter_df["chi"] = chis
            iter_df_list.append(iter_df)

            # Calc the current best params and set the params range for next iteration
            min_chi_index = np.argmin(chis)
            new_min_chi = chis[min_chi_index]
            min_params_comb = params_combinations[min_chi_index]
            curr_params_range = [curr_params_range[i].get_new_range(min_param) for i, min_param in enumerate(min_params_comb)]
            
            counter += 1

        # print(f"finished!! counter: {counter} | chi: {min_chi} | min_comb: {min_params} \n")
        df_params_comb = pd.concat(iter_df_list, keys=range(len(iter_df_list)))
        df_params_comb["delta_chi"] = df_params_comb["chi"] - min_chi

        return min_params, min_chi, df_params_comb        

    def _calc_chi_on_params_meshgrid(self, params: List[ParamRange], fixed_params: FloatDict) -> Tuple[List[float], npt.ArrayLike]:

        # create a meshgrid of all possible params range combinations
        params_combinations = np.array(np.meshgrid(*[param.range() for param in params])).T.reshape(-1,len(params))

        # Calculate the chi sq value for each combination
        chis = []
        for comb in params_combinations:
            # Gather the param combination in a dictionary
            comb_dict = {params[i].name: param for i, param in enumerate(comb)}
            # Calc the chi sq value corresponding to the current param combination
            chis.append(calc_chi_sq(self.x, self.y, self.y_err, comb_dict, fixed_params, self.target_func))

        #print(f"{len(params_combinations)} combs took {end_time-start_time}")
        return chis, params_combinations

    def _fit_params_errors(self, df_params_for_e: pd.DataFrame, fit_params: FloatDict, confidence_level: float = 68.3) -> ValErrDict:
        chi_confidence = CONFIDENCE_TO_DELTA_CHI_BY_DDOF[1][confidence_level]
        
        df = df_params_for_e[(df_params_for_e['delta_chi'] <= chi_confidence * 1.2) & (df_params_for_e['delta_chi'] >= chi_confidence)]
        df_max = df.max()
        df_min = df.min()
        
        fit_params_with_errors: ValErrDict = {}
        for name, val in fit_params.items():
            upper_error = abs(df_max[name] - val)
            lower_error = abs(val - df_min[name])
            fit_params_with_errors[name] = ValueWithError(f"{name}_non_linear_{len(fit_params)}_params", val, upper_error, lower_error)

        return fit_params_with_errors

    def _fit_params_errors2(self, df_params_for_e: pd.DataFrame, fit_params: FloatDict) -> ValErrDict:
        df = df_params_for_e[df_params_for_e['delta_chi'] >= 1]
        fit_params_with_errors: ValErrDict = {}
        for p_index, val in fit_params.items():
            temp_df=df
            big_df = temp_df[temp_df[p_index]>=fit_params[p_index]]
            small_df = temp_df[temp_df[p_index]<=fit_params[p_index]]
            min_bound = small_df[small_df['delta_chi']==small_df['delta_chi'].min()].reset_index()[p_index][0]
            print(small_df['delta_chi'].min())
            max_bound = big_df[big_df['delta_chi']==big_df['delta_chi'].min()].reset_index()[p_index][0]
            print(big_df['delta_chi'].min())
            down_error = fit_params[p_index] - min_bound
            up_error = max_bound - fit_params[p_index]
            fit_params_with_errors[p_index] = ValueWithError(f"{p_index}_non_linear_{len(fit_params)}_params", val, up_error, down_error)

        return fit_params_with_errors

    def plot_fit(self, params: FloatDict):
        f_x = self.target_func(params, self.x)
        fit_plot(xlabel=TIME, ylabel=I_VAL, x=self.x, y=self.y, y_error=self.y_err, y_est=f_x)
        residue_plot(xlabel=TIME, ylabel=I_VAL, x=self.x, y=self.y, y_error=self.y_err, y_est=f_x)

    def plot_2d_contours(self, ax: Axes, x_param_name: str, y_param_name: str, fixed_params: FloatDict):
        if self.df_params_comb is None:
            print("You must fit before plotting contours!")
            return

        print(f'{x_param_name} vs. {y_param_name} effect on Goodness of Fit (chi)')
        levels = list(CONFIDENCE_TO_DELTA_CHI_BY_DDOF[2].values())

        fixed_params = fixed_params | {name: param.value for name, param in self.fit_params.items() if name != x_param_name and name != y_param_name}
        df_x_y_params = self._chis_for_contours(fit_params=[x_param_name, y_param_name], fixed_params=fixed_params)
        
        #cond = delta_chis < np.max(levels) * 2
        #iter = delta_chis[cond].idxmax()[0]+1
        #param_x_iter, param_y_iter, delta_chis_iter = param_x[iter], param_y[iter], delta_chis[iter]
        x_param = df_x_y_params[x_param_name]
        y_param = df_x_y_params[y_param_name]
        delta_chis = df_x_y_params["delta_chi"]
        
        cs = ax.tricontour(x_param, y_param, delta_chis, levels=levels, linewidths=0.5,colors=('red',  'green', 'orange'))

        # Fit figure to contours
        contour_points = cs.collections[len(levels)-1].get_paths()[0].vertices
        contour_x = contour_points[:,0]
        contour_y = contour_points[:,1]
        x_boundry = (max(contour_x) - min(contour_x)) / 10
        y_boundry = (max(contour_y) - min(contour_y)) / 10
        ax.set_xlim(min(contour_x)-x_boundry, max(contour_x)+x_boundry)
        ax.set_ylim(min(contour_y)-y_boundry, max(contour_y)+y_boundry)
        # ax2.set_aspect('equal')

        ax.clabel(cs, fmt=DELTA_CHI_TO_CONFIDENCE_BY_DDOF[2], inline=True, fontsize=10)

        # Plot the best param center point
        fit_x = self.fit_params[x_param_name]
        fit_y = self.fit_params[y_param_name]
        ax.scatter(fit_x.value,fit_y.value)
        ax.annotate(f"({fit_x.str_value()},{fit_y.str_value()})", (fit_x.value,fit_y.value), textcoords="offset points", xytext=(0,10), ha='center')
        
        #ax2.scatter(x=param_x, y=param_y, c=delta_chis)
        #cs = ax2.tricontour(u_min,T0,chis, levels=levels, linewidths=0.5,colors=('red',  'green', 'orange'))
        #cntr2 = ax2.tricontourf(u_min,T0,chis, levels=levels, cmap='Blues')

        #fig.colorbar(cntr2, ax=ax2)
        #ax2.plot(u_min,T0, 'ko', ms=1)
        #ax2.set_title(f'{x_param_name} vs. {y_param_name} effect on Goodness of Fit (chi)')
        #ax.set_xlabel(LABELS.get(x_param_name, x_param_name), fontsize=10)
        #ax.set_ylabel(LABELS.get(y_param_name, y_param_name), fontsize=10)
        ax.grid()
        #plt.savefig(f"contour_{len(self.fit_params)}D_{x_param_name}_vs_{y_param_name}_{datetime.datetime.now()}.pdf")
        #plt.show()

    def _chis_for_contours(self, fit_params: List[str], fixed_params: FloatDict) -> pd.DataFrame:
        delta_chi_bound = max(CONFIDENCE_TO_DELTA_CHI_BY_DDOF[2].values()) * 10

        df_bound = self.df_params_comb[self.df_params_comb['delta_chi'] < delta_chi_bound]
        param_upper_bounds = df_bound.max()
        param_lower_bounds = df_bound.min()
        params_range = [ParamRange(name, param_lower_bounds[name], param_upper_bounds[name], 200) for name in fit_params]

        chis, params_combinations = self._calc_chi_on_params_meshgrid(params_range, fixed_params)
        df_params_for_contour = pd.DataFrame(data=params_combinations, columns=fit_params)
        df_params_for_contour["delta_chi"] = np.array(chis) - self.fit_chi
        
        return df_params_for_contour
    


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

    var_y_inv = np.diag(1 / (y_err ** 2))           # V^-1 - the inv var matrix of y
    inter_res = C.T @ var_y_inv                     # intermediate result - C^T * V^-1
    var_param_est = np.linalg.inv(inter_res @ C)    # The params var matrix - (C^T * V^-1 * C)^-1
    pararm_est = var_param_est @ inter_res @ y      # The params vector - (C^T * V^-1 * C)^-1 * C^T * V^-1 * y
    y_est = C @ pararm_est                          # The calculated y=Ca

    # calc chi sq red
    residue_mat = y - y_est
    ddof = max(len(x) - n_param, 1)
    chi_sq_red = residue_mat.T @ var_y_inv @ residue_mat / ddof
    
    return pararm_est, np.diag(var_param_est) ** 0.5, y_est, chi_sq_red

########################################## Calculation Functions ##########################################

def calc_chi_sq(x: npt.ArrayLike, y: npt.ArrayLike, y_error: npt.ArrayLike, params: FloatDict, fixed_params: FloatDict, target_func: TargetFunc) -> float:
    return sum_of_sq((target_func(params | fixed_params, x) - y) / y_error)

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

def nsigma(expected: ValueWithError, meas: ValueWithError) -> float:
    """Calculate the n-sigma test between measured and expected values

    Args:
        expected (`value_with_error`): The expected value and its error
        meas (`value_with_error`): The measured value and its error

    Returns:
        float: n-sigma test
    """
    if expected.value > meas.value:
        return (expected.value-meas.value) / sqrt_sum_of_sq([expected.lower_error, meas.upper_error])
    elif expected.value < meas.value:
        return (meas.value-expected.value) / sqrt_sum_of_sq([expected.upper_error, meas.lower_error])
    else:
        return 0

def error_combination(derivative: npt.ArrayLike, error: npt.ArrayLike) -> float:
    """Calculate the error combination of ind. errors

    Args:
        derivative (`ArrayLike`): The calculated partial derivative W.R.T to the errors (weights)
        error (`ArrayLike`): The errors to be combined

    Returns:
        float: The error combination
    """
    return sqrt_sum_of_sq(np.multiply(derivative, error))



########################################## Plot & Print Functions ##########################################

def fit_plot(xlabel: str, ylabel: str, x: npt.ArrayLike, y: npt.ArrayLike, y_error: npt.ArrayLike, y_est: npt.ArrayLike):
    plt.errorbar(x=x, y=y, yerr=y_error, fmt='o', markersize=2)
    plt.plot(x, y_est)
    plt.xlabel(LABELS.get(xlabel, xlabel))
    plt.ylabel(LABELS.get(ylabel, ylabel))
    plt.grid()
    plt.legend(["Fit line", "Observations"])
    plt.tight_layout()
    plt.savefig(f"fit_{xlabel}_vs_{ylabel}_{datetime.datetime.now()}.pdf")
    plt.show()

def residue_plot(xlabel: str, ylabel: str, x: npt.ArrayLike, y: npt.ArrayLike, y_error: npt.ArrayLike, y_est: npt.ArrayLike):
    plt.errorbar(x=x, y=y-y_est, yerr=y_error, fmt='o', markersize=2)
    plt.xlabel(LABELS.get(xlabel, xlabel))
    plt.ylabel(f"Residue {LABELS.get(ylabel, ylabel)}")
    plt.axhline(y = 0, linestyle = '--')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"residue_{xlabel}_vs_{ylabel}_{datetime.datetime.now()}.pdf")
    plt.show()

def norm_hist(name: str, data: npt.ArrayLike) -> ValueWithError:
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the histogram
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g', ec='black')
    plt.xlabel(LABELS.get(name, name))
    plt.ylabel("Probability Density")

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    #title = "Fit results: mu = %.4f,  std = %.4f" % (mu, std)
    #plt.title(title)
    plt.legend([f"Normal Distribution", "Histogram"])
    plt.tight_layout()
    plt.savefig(f"hist_{name}_{datetime.datetime.now()}.pdf")
    plt.show()

    return ValueWithError(name+"_hist", mu, std)

def bootstrap_compare(fit: ValueWithError, hist: ValueWithError) -> float:
    res = np.abs(fit.value-hist.value) / fit.value
    
    print(fit)
    print(hist)
    print(f"bootstrap value comparison: {res}")
    print(f"bootstrap error comparison: fit error order e{orderOfMagnitude(fit.upper_error)}, bootstrap error order e{orderOfMagnitude(hist.upper_error)}")
    print(f"Nsigma sanity check: {nsigma(fit, hist)}")

    return res

def calc_I(params: FloatDict, x: npt.ArrayLike) -> npt.ArrayLike:
    u_min = params[U_MIN]
    t0 = params[T0]
    tau = params[TAU]
    f_bl = params[F_BL]
    u_t = np.sqrt(u_min ** 2 + ((x-t0)/tau) ** 2)
    mu = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    return f_bl * (mu - 1) + 1