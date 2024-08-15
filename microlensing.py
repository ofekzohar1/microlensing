import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from ms_helper import value_with_error
import ms_helper as msh

class microlensing:
    def __init__(self, event_url: str) -> None:
        self.data = pd.read_csv(event_url+"/phot.dat", sep='\s+', comment='#', usecols=range(3), names=["JHD", "magnitude", "magnitude_error"])
        self.param_pd = pd.read_csv(event_url+"/params.dat", sep='\s+', comment='#', index_col=0, skiprows=8, names=["param", "value", "error"])
        self._ogle_process()
        self._data_process()
        self.par_params = {}
            
    def _ogle_process(self) -> None:
        self.ogle = {name: value_with_error(name, row["value"], row["error"]) for name, row in self.param_pd.iterrows()}
        self.ogle["t0"] = self.ogle.pop("Tmax")
        self.ogle["m0"] = self.ogle.pop("I0")
        self.ogle["m_bl"] = self.ogle.pop("I_bl")
        self.ogle["Imax"] = self.ogle.pop("Amax")
        for name, value in self.ogle.items():
            value.name = name + "_ogle"

    def _data_process(self) -> None:
        self.data["norm_time"] = self.data["JHD"] - np.min(self.data["JHD"])
        m0_ogle = self.ogle["m0"]
        self.data["I"] = 10 ** (-0.4 * (self.data["magnitude"]-m0_ogle.value))
        self.data["I_error"] = 0.4 * np.log(10) * self.data["I"] * np.sqrt(self.data["magnitude_error"] ** 2 + m0_ogle.error ** 2)

    def parabolic_fit(self, mid_range: float, range_len: float) -> Dict[str, value_with_error]:
        min_range, max_range = mid_range-range_len, mid_range+range_len
        data_cut = self.data[(self.data["JHD"]<=max_range) & (self.data["JHD"]>=min_range)]
        a, std_a, est_for_x, chi = msh.independent_meas_linear_fit(n_param=3, x=data_cut["JHD"]-min_range, y=data_cut["I"], y_error=data_cut["I_error"])

        t0_par, Imax_par = self._extract_parabolic_params(a, std_a, min_range)
        self.par_params["t0"] = t0_par
        self.par_params["Imax"] = Imax_par

        print(f"Chi: {chi}")
        # plot
        plt.errorbar(x=data_cut["JHD"], y=data_cut["I"], yerr=data_cut["I_error"], fmt='o', markersize=2)
        plt.plot(data_cut["JHD"], est_for_x)
        plt.grid()
        plt.show()

        # plot residuals
        msh.residue_plot(xlabel="time [JHD]", ylabel="Residue I", x=data_cut["JHD"], y=data_cut["I"], y_error=data_cut["I_error"], y_est=est_for_x)

        # nsigma for ogle
        print("nsigma with ogle params:")
        print(self.ogle['t0'])
        print(self.par_params['t0'])
        print(f"nsigma: {msh.nsigma(self.ogle['t0'], self.par_params['t0'])}")
        print()
        print(self.ogle['Imax'])
        print(self.par_params['Imax'])
        print(f"nsigma: {msh.nsigma(self.ogle['Imax'], self.par_params['Imax'])}")

        self.data_cut=data_cut
        return self.par_params
            

    def _extract_parabolic_params(self, a: npt.ArrayLike, std_a: npt.ArrayLike, time_fix: float) -> Tuple[value_with_error, value_with_error]:
        # t0_par calc
        t0_par_value = -0.5 * a[1] / a[2] + time_fix
        t0_par_error = msh.error_combination([0.5/a[2], 0.5 * a[1]/(a[2]**2)], std_a[1:])
        t0_par = value_with_error("t0_par", t0_par_value, t0_par_error)

        # Imax_par calc
        Imax_par_value = a[0] - 0.25 * (a[1] ** 2) / a[2]
        Imax_par_error = msh.error_combination([1, 0.5*a[1]/a[2], 0.25*(a[1]/a[2])**2], std_a)
        Imax_par = value_with_error("Imax_par", Imax_par_value, Imax_par_error)

        return t0_par, Imax_par

    def bootstrap(self, mid_range: float, range_len: float, iter: int=10000) -> None:
        min_range, max_range = mid_range-range_len, mid_range+range_len
        data_cut = self.data[(self.data["JHD"]<=max_range) & (self.data["JHD"]>=min_range)]

        t0_list, Imax_list = [], []
        for _ in range(iter):
            sample = data_cut.sample(n=len(data_cut), replace=True)
            a, std_a, _, _ = msh.independent_meas_linear_fit(n_param=3, x=sample["JHD"]-min_range, y=sample["I"], y_error=sample["I_error"])
            t0_par, Imax_par = self._extract_parabolic_params(a, std_a, min_range)
            t0_list.append(t0_par.value)
            Imax_list.append(Imax_par.value)

        t0_hist = msh.norm_hist("t0", t0_list)
        Imax_hist = msh.norm_hist("Imax", Imax_list)

        # compare to the original fit
        print("bootstrap:")
        msh.bootstrap_compare(self.par_params["t0"], t0_hist)
        print()

        msh.bootstrap_compare(self.par_params["Imax"], Imax_hist)
                
       