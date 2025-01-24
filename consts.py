############################################### Consts ###############################################

CONFIDENCE_TO_DELTA_CHI_BY_DDOF =[
    {},
    {68.3: 1.0, 90: 2.71, 95.4: 4.0, 99: 6.63},
    {68.3: 2.3, 90: 4.61, 95.4: 6.17, 99: 9.21},
    {68.3: 3.53, 90: 6.25, 95.4: 8.02, 99: 11.3},
    ]

U_MIN = "umin"
T0 = "t0"
F_BL = "fbl"
TAU = "tau"
I_MAX = "Imax"
M0 = "m0"
M_BL = "m_bl"
JHD = "JHD"
NORM_TIME = "norm_time"
TIME = "time"
I_VAL = "I"
I_ERROR = I_VAL+"_error"

LABELS = {
    T0: f"T0 [{JHD}]",
    TAU: f"Tau [{JHD}]",
    TIME: f"Time [{JHD}]",
    NORM_TIME: f"Normalized Time [{JHD}]",
    I_VAL: "Normalized Intensity"
}