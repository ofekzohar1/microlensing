import time
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import utils as ms_utils
from utils import ValueWithError, FloatDict, ParamRange, ValErrDict
from consts import *

class Microlensing:
    def __init__(self, event_url: str) -> None:
        self.data = pd.read_csv(event_url+"/phot.dat", sep='\s+', comment='#', usecols=range(3), names=[JHD, "magnitude", "magnitude_error"])
        self.param_pd = pd.read_csv(event_url+"/params.dat", sep='\s+', comment='#', index_col=0, skiprows=8, names=["param", "value", "error"])
        self._ogle_process()
        self._data_process()
        self.par_params: ValErrDict = {}
        self.non_linear_params: ValErrDict = {}
        self.non_lin_fit: ms_utils.MeshgridChiMinNonLinearFit = None
            
    def _ogle_process(self) -> None:
        self.ogle = {name: ValueWithError(name, row["value"], row["error"]) for name, row in self.param_pd.iterrows()}
        self.ogle[T0] = self.ogle.pop("Tmax")
        self.ogle[M0] = self.ogle.pop("I0")
        self.ogle[M_BL] = self.ogle.pop("I_bl")
        self.ogle[I_MAX] = self.ogle.pop("Amax")
        for name, value in self.ogle.items():
            value.name = name + "_ogle"

    def _data_process(self) -> None:
        self.data[NORM_TIME] = self.data[JHD] - np.min(self.data[JHD])
        m_bl_ogle = self.ogle[M_BL]
        self.data[I_VAL] = 10 ** (-0.4 * (self.data["magnitude"]-m_bl_ogle.value))
        self.data[I_ERROR] = 0.4 * np.log(10) * self.data[I_VAL] * np.sqrt(self.data["magnitude_error"] ** 2 + m_bl_ogle.upper_error ** 2)

    def parabolic_fit(self, mid_range: float, range_len: float) -> ValErrDict:
        min_range, max_range = mid_range-range_len, mid_range+range_len
        data_cut = self.data[(self.data[JHD]<=max_range) & (self.data[JHD]>=min_range)]
        a, std_a, est_for_y, chi = ms_utils.independent_meas_linear_fit(n_param=3, x=data_cut[JHD]-min_range, y=data_cut[I_VAL], y_err=data_cut[I_ERROR])

        t0_par, Imax_par, umin_par = self._extract_parabolic_params(a, std_a, min_range)
        self.par_params[T0] = t0_par
        self.par_params[I_MAX] = Imax_par
        self.par_params[U_MIN] = umin_par

        # plot fit and residuals
        ms_utils.fit_plot(xlabel=TIME, ylabel=I_VAL, x=data_cut[JHD], y=data_cut[I_VAL], y_error=data_cut[I_ERROR], y_est=est_for_y)
        ms_utils.residue_plot(xlabel=TIME, ylabel=I_VAL, x=data_cut[JHD], y=data_cut[I_VAL], y_error=data_cut[I_ERROR], y_est=est_for_y)

        print(f"Chi: {chi}")
        print()

        # nsigma for ogle
        print("nsigma with ogle params:")
        print(self.ogle[T0])
        print(self.par_params[T0])
        print(f"nsigma: {ms_utils.nsigma(self.ogle[T0], self.par_params[T0])}")
        print()
        print(self.ogle['Imax'])
        print(self.par_params['Imax'])
        print(f"nsigma: {ms_utils.nsigma(self.ogle['Imax'], self.par_params['Imax'])}")
        print()
        print(self.ogle[U_MIN])
        print(self.par_params[U_MIN])
        print(f"nsigma: {ms_utils.nsigma(self.ogle[U_MIN], self.par_params[U_MIN])}")

        self.data_cut=data_cut
        return self.par_params

    def non_linear_fit(self, init_params: List[ParamRange], fixed_params: FloatDict, res_chi: float = 0, max_iters: int = np.inf) -> ValErrDict:
        self.non_lin_fit = ms_utils.MeshgridChiMinNonLinearFit(self.data[JHD], self.data[I_VAL], self.data[I_ERROR], Microlensing.calc_I)
        self.non_linear_params, fit_chi  = self.non_lin_fit.fit(init_params, fixed_params, max_iters=max_iters)

        # plot fit and residuals
        dict_fit_params = {name: param.value for name, param in self.non_linear_params.items()}
        full_min_params = dict_fit_params | fixed_params
        self.non_lin_fit.plot_fit(full_min_params)

        ddof = max(len(self.non_lin_fit.x) - len(init_params), 1)
        print(f"Chi: {fit_chi} (Chi Reduced: {fit_chi / ddof})")
        print()

        # nsigma for ogle
        print("--- nsigma with ogle params ---")
        for name, param in self.non_linear_params.items():
            ogle_param = self.ogle[name]
            print(ogle_param)
            print(param)
            print(f"nsigma: {ms_utils.nsigma(ogle_param, param)}")
            print()

        print("--- nsigma with parabolic params ---")
        for name, param in self.non_linear_params.items():
            if name in self.par_params:
                print(self.par_params[name])
                print(param)
                print(f"nsigma: {ms_utils.nsigma(self.par_params[name], param)}")
                print()

        return self.non_linear_params

    def non_linear_contours(self, ParamList: List[str]):
        for i, param1 in enumerate(ParamList):
            for param2 in ParamList[i+1:]:
                self.non_lin_fit.plot_2d_contours(param1, param2)
            

    def _extract_parabolic_params(self, a: npt.ArrayLike, std_a: npt.ArrayLike, time_fix: float) -> Tuple[ValueWithError, ValueWithError, ValueWithError]:
        # t0_par calc
        t0_par_value = -0.5 * a[1] / a[2] + time_fix
        t0_par_error = ms_utils.error_combination([0.5/a[2], 0.5 * a[1]/(a[2]**2)], std_a[1:])
        t0_par = ValueWithError(f"{T0}_par", t0_par_value, t0_par_error)

        # Imax_par calc
        Imax_par_value = a[0] - 0.25 * (a[1] ** 2) / a[2]
        Imax_par_error = ms_utils.error_combination([1, 0.5*a[1]/a[2], 0.25*(a[1]/a[2])**2], std_a)
        Imax_par = ValueWithError(f"{I_MAX}_par", Imax_par_value, Imax_par_error)

        # umin_par calc
        mu_par = Microlensing.mu_from_I_and_fbl(Imax_par, self.ogle[F_BL])
        umin_par = Microlensing.u_min_from_mu_max(mu_par)
        umin_par.name = f"{U_MIN}_par"

        return t0_par, Imax_par, umin_par

    def non_lin_bootstrap(self, init_params: List[ParamRange], fixed_params: FloatDict, res_chi: float = 0, max_mesh_iters: int = np.inf, iter: int=10000) -> None:
        param_list_values: Dict[str, List[float]] = {}
        old_tick = 0
        for i in range(iter):
            if i % 100 == 0:
                new_tick = time.time()
                print(i, f"took {new_tick-old_tick}")
                old_tick = new_tick

            sample = self.data.sample(n=len(self.data), replace=True)
            non_lin_fit_iter = ms_utils.MeshgridChiMinNonLinearFit(sample[JHD], sample[I_VAL], sample[I_ERROR], Microlensing.calc_I)
            fit_params, _, _ = non_lin_fit_iter._meshgrid_fit(init_params, fixed_params, res_chi, max_mesh_iters)
            for name, val in fit_params.items():
                if name not in param_list_values:
                    param_list_values[name] = []
                param_list_values[name].append(val)

        histogram_dict: ValErrDict = {}
        for name, val_list in param_list_values.items():
            histogram_dict[name] = ms_utils.norm_hist(name, val_list)

        # compare to the original fit
        print("bootstrap:")
        for name, hist_val in histogram_dict.items():
            ms_utils.bootstrap_compare(self.non_linear_params[name], hist_val)
            print()

    def bootstrap(self, mid_range: float, range_len: float, iter: int=10000) -> None:
        min_range, max_range = mid_range-range_len, mid_range+range_len
        data_cut = self.data[(self.data[JHD]<=max_range) & (self.data[JHD]>=min_range)]

        t0_list, Imax_list, umin_list = [], [], []
        for _ in range(iter):
            sample = data_cut.sample(n=len(data_cut), replace=True)
            fit_params, std_a, _, _ = ms_utils.independent_meas_linear_fit(n_param=3, x=sample[JHD]-min_range, y=sample[I_VAL], y_err=sample[I_ERROR])
            t0_par, Imax_par, umin_par = self._extract_parabolic_params(fit_params, std_a, min_range)
            t0_list.append(t0_par.value)
            Imax_list.append(Imax_par.value)
            umin_list.append(umin_par.value)

        t0_hist = ms_utils.norm_hist(T0, t0_list)
        Imax_hist = ms_utils.norm_hist(I_MAX, Imax_list)
        umin_hist = ms_utils.norm_hist(U_MIN, umin_list)

        # compare to the original fit
        print("bootstrap:")
        ms_utils.bootstrap_compare(self.par_params[T0], t0_hist)
        print()
        ms_utils.bootstrap_compare(self.par_params[I_MAX], Imax_hist)
        print()
        ms_utils.bootstrap_compare(self.par_params[U_MIN], umin_hist)

    ########################################## Class Static Functions ##########################################
        
    def calc_I(params: FloatDict, x: npt.ArrayLike) -> npt.ArrayLike:
        u_min = params[U_MIN]
        t0 = params[T0]
        tau = params[TAU]
        f_bl = params[F_BL]
        u_t = np.sqrt(u_min ** 2 + ((x-t0)/tau) ** 2)
        mu = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
        return f_bl * (mu - 1) + 1

    def mu_from_I_and_fbl(I: ValueWithError, fbl: ValueWithError) -> ValueWithError:
            mu_value = (I.value-1) / fbl.value + 1
            mu_derivative_wrt_I = 1 / fbl.value
            mu_derivative_wrt_fbl = (I.value-1) / (fbl.value ** 2)
            mu_error = ms_utils.error_combination([mu_derivative_wrt_I, mu_derivative_wrt_fbl], [I.upper_error, fbl.upper_error])

            return ValueWithError("mu", mu_value, mu_error)

    def u_min_from_mu_max(mu: ValueWithError) -> ValueWithError:
            u_value = np.sqrt(2 * (mu.value / np.sqrt(mu.value**2 - 1) - 1))
            
            u_derivative_wrt_mu = 1 / (u_value * ((mu.value**2 - 1) ** 1.5))
            u_error = ms_utils.error_combination([u_derivative_wrt_mu], [mu.upper_error])

            return ValueWithError(U_MIN, u_value, u_error)