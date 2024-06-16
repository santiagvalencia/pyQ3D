import aerosandbox as asb
import numpy as np


def get_mean_sweep(
    sec1: asb.WingXSec, sec2: asb.WingXSec, x_nondim=0.25, radians=False
):

    if not 0 <= x_nondim <= 1:
        raise ValueError(
            "x_nondim (the nondimensional chord) should be between 0 and 1."
        )

    wing = asb.Wing(xsecs=[sec1, sec2])

    sweep_angle = wing.mean_sweep_angle(x_nondim=x_nondim)

    if radians:
        return np.deg2rad(sweep_angle)

    return sweep_angle


def find_interpolated_airfoil(): ...
