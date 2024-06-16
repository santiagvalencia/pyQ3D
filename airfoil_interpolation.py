import aerosandbox as asb
import numpy as np
from scipy.interpolate import interp1d


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


def get_xyz_le_from_xsec(sec: asb.WingXSec):
    return tuple(sec.xyz_le)


def get_spanwise_equal_chord_line(
    x_1,
    y_1,
    x_2,
    y_2,
    inverse=False,
    return_as_dict=False,
):
    slope = (y_2 - y_1) / (x_2 - x_1)
    intercept = y_1 - slope * x_1

    if inverse:
        intercept = -intercept / slope
        slope = 1 / slope

    if return_as_dict:
        return {"slope": slope, "intercept": intercept}

    return lambda x: slope * x + intercept


def get_line_intersection(dict_1, dict_2):
    m_1 = dict_1["slope"]
    b_1 = dict_1["intercept"]

    m_2 = dict_2["slope"]
    b_2 = dict_2["intercept"]

    x = (b_1 - b_2) / (m_2 - m_1)
    y = (b_1 * m_2 - b_2 * m_1) / (m_2 - m_1)

    return x, y


def get_perp_line(y_nondim, sec1: asb.WingXSec, sec2: asb.WingXSec, xi_c=0.25):

    x_root, y_root, _ = get_xyz_le_from_xsec(sec1)
    x_tip, y_tip, _ = get_xyz_le_from_xsec(sec2)

    span = y_tip - y_root

    # find leading edge point of interest
    y_le = y_nondim * span

    f_le = get_spanwise_equal_chord_line(x_root, y_root, x_tip, y_tip, inverse=True)

    x_le = f_le(y_le)

    # find quarter-chord line
    lambda_qc = get_mean_sweep(sec1, sec2, xi_c, radians=True)
    dict_perp_qc = {
        "slope": -np.tan(lambda_qc),
        "intercept": -np.tan(lambda_qc) * -x_le + y_le,
    }

    # find trailing edge line
    x_te_root = x_root + sec1.chord
    x_te_tip = x_tip + sec2.chord

    y_te_root = y_root
    y_te_tip = y_tip

    dict_te = get_spanwise_equal_chord_line(
        x_te_root, y_te_root, x_te_tip, y_te_tip, return_as_dict=True
    )

    # find trailing edge point of interest
    x_te, y_te = get_line_intersection(dict_te, dict_perp_qc)

    chord_perp = np.sqrt((x_le - x_te) ** 2 + (y_le - y_te) ** 2)

    return {
        "line_dict": dict_perp_qc,
        "x_te": x_te,
        "y_te": y_te,
        "x_le": x_le,
        "y_le": y_le,
        "chord_perp": chord_perp,
        "lambda_qc": lambda_qc,
    }


def get_interpolation_weights(y_nondim, x_nondims, sec1, sec2, xi_c=0.25):

    data = get_perp_line(y_nondim, sec1, sec2, xi_c=xi_c)

    x_le_root, y_le_root, _ = get_xyz_le_from_xsec(sec1)
    x_le_tip, y_le_tip, _ = get_xyz_le_from_xsec(sec2)

    weights_tip = []
    weights_root = []

    x_nondims_geom_axes = np.array(x_nondims) * np.cos(data["lambda_qc"])

    for x_nondim in x_nondims_geom_axes:
        x_root = x_le_root + x_nondim * sec1.chord
        x_tip = x_le_tip + x_nondim * sec2.chord

        line = get_spanwise_equal_chord_line(
            x_root, y_le_root, x_tip, y_le_tip, return_as_dict=True
        )

        x_i, y_i = get_line_intersection(data["line_dict"], line)

        w_tip = y_i / (y_le_tip - y_le_root)
        w_root = 1 - w_tip

        weights_tip.append(w_tip)
        weights_root.append(w_root)

    return np.array(weights_root), np.array(weights_tip)


def interpolate_airfoil_chordwise(af: asb.Airfoil):

    top_function = interp1d(
        af.upper_coordinates()[:, 0],
        af.upper_coordinates()[:, 1],
        kind="cubic",
        fill_value="extrapolate",
    )

    bottom_function = interp1d(
        af.lower_coordinates()[:, 0],
        af.lower_coordinates()[:, 1],
        kind="cubic",
        fill_value="extrapolate",
    )

    return top_function, bottom_function


def find_interpolated_airfoil(
    y_nondim, x_nondims, sec1: asb.WingXSec, sec2: asb.WingXSec, xi_c=0.25
):

    weights_root, weights_tip = get_interpolation_weights(
        y_nondim, x_nondims, sec1, sec2, xi_c=xi_c
    )

    weights_root_extended = np.concatenate((weights_root, weights_root[::-1]))
    weights_tip_extended = np.concatenate((weights_tip, weights_tip[::-1]))

    af_1 = sec1.airfoil
    af_2 = sec2.airfoil

    af_1_top, af_1_bottom = interpolate_airfoil_chordwise(af_1)
    af_2_top, af_2_bottom = interpolate_airfoil_chordwise(af_2)

    af_1_interpolated_z = np.concatenate(
        (af_1_top(x_nondims), af_1_bottom(x_nondims)[::-1])
    )

    af_2_interpolated_z = np.concatenate(
        (af_2_top(x_nondims), af_2_bottom(x_nondims)[::-1])
    )

    new_af_z = (
        af_1_interpolated_z * weights_root_extended
        + af_2_interpolated_z * weights_tip_extended
    )
    new_af_x = np.concatenate((x_nondims, x_nondims[::-1]))

    new_airfoil = asb.Airfoil(coordinates=np.column_stack((new_af_x, new_af_z)))

    return new_airfoil
