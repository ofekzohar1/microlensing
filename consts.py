############################################### Consts ###############################################


from typing import Dict, List


CONFIDENCE_TO_DELTA_CHI_BY_DDOF: List[Dict[float, float]] = [
    {},
    {68.3: 1.0, 95.4: 4.0, 99.73: 9.0},
    {68.3: 2.3, 95.4: 6.17, 99.73: 11.8},
    {68.3: 3.53, 95.4: 8.02, 99.73: 14.2},
    ]

DELTA_CHI_TO_CONFIDENCE_BY_DDOF: List[Dict[float, str]] = [
    {},
    {1.0: "68.3%", 4.0: "95.4%", 9.0: "99.73%"},
    {2.3: "68.3%", 6.17: "95.4%", 11.8: "99.73%"},
    {3.53: "68.3%",8.02: "95.4%", 14.2: "99.73%"},
    ]

U_MIN = "umin"
T0 = "t0"
F_BL = "fbl"
TAU = "tau"
I_MAX = "Imax"
M0 = "m0"
M_BL = "m_bl"
JHD = "JHD"
HJD = "HJD"
NORM_TIME = "norm_time"
TIME = "time"
I_VAL = "I"
I_ERROR = I_VAL+"_error"

LABELS = {
    T0: r"Time of Maximal Approach $t_{0}$" + f" [{HJD}]",
    TAU: r"Einstein-crossing Timescale $\tau$" + f" [{HJD}]",
    TIME: f"Time [{HJD}]",
    NORM_TIME: f"Normalized Time [{HJD}]",
    I_VAL: r"Normalized Intensity $\mathtt{I/I^*}$",
    U_MIN: r"Normalized Einstein-impact-angle $u_{min}$",
    F_BL: r"Blending Light Ratio $f_{bl}$"
}