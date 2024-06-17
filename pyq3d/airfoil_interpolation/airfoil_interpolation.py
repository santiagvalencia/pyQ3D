from typing import Callable, Dict, List, Tuple, Union

import aerosandbox as asb
import numpy as np
from scipy.interpolate import interp1d

__all__ = [
    "calculate_mean_sweep_angle",
    "get_xyz_le_from_xsec",
    "get_spanwise_equal_chord_line",
    "get_line_intersection",
    "get_perp_line",
    "get_interpolation_weights",
    "interpolate_airfoil",
    "interpolate_airfoil_chordwise",
    "find_interpolated_airfoil",
]


def calculate_mean_sweep_angle(
    sec1: asb.WingXSec,
    sec2: asb.WingXSec,
    x_nondim: float = 0.25,
    radians: bool = False,
) -> float:
    """
    Calculate the mean sweep angle between two wing sections.

    Args:
        sec1 (asb.WingXSec): The root wing section.
        sec2 (asb.WingXSec): The tip wing section.
        x_nondim (float, optional): The nondimensional chord position. Defaults to 0.25.
        radians (bool, optional): Return the sweep angle in radians if True. Defaults to False.

    Returns:
        float: The mean sweep angle.
    """
    if not 0 <= x_nondim <= 1:
        raise ValueError(
            "x_nondim (the nondimensional chord) should be between 0 and 1."
        )

    wing = asb.Wing(xsecs=[sec1, sec2])
    sweep_angle = wing.mean_sweep_angle(x_nondim=x_nondim)
    return np.deg2rad(sweep_angle) if radians else sweep_angle


def get_xyz_le_from_xsec(sec: asb.WingXSec) -> Tuple[float]:
    """
    Extract leading edge coordinates from a wing section.

    Args:
        sec (asb.WingXSec): The wing section.

    Returns:
        tuple: The (x, y, z) coordinates of the leading edge.
    """
    return tuple(sec.xyz_le)


def get_spanwise_equal_chord_line(
    x_1: float,
    y_1: float,
    x_2: float,
    y_2: float,
    inverse: bool = False,
    return_as_dict: bool = False,
) -> Union[Callable[[float], float], Dict[str, float]]:
    """
    Calculate the spanwise equal chord line between two points.

    Args:
        x_1 (float): x-coordinate of the first point.
        y_1 (float): y-coordinate of the first point.
        x_2 (float): x-coordinate of the second point.
        y_2 (float): y-coordinate of the second point.
        inverse (bool, optional): Return the inverse (x = x(y)) line. Defaults to False.
        return_as_dict (bool, optional): Return as a dictionary. Defaults to False.

    Returns:
        function or dict: The spanwise equal chord line as a function or dictionary.
    """
    slope = (y_2 - y_1) / (x_2 - x_1)
    intercept = y_1 - slope * x_1

    if inverse:
        intercept = -intercept / slope
        slope = 1 / slope

    if return_as_dict:
        return {"slope": slope, "intercept": intercept}

    return lambda x: slope * x + intercept


def get_line_intersection(
    dict_1: Dict[str, float], dict_2: Dict[str, float]
) -> Tuple[float]:
    """
    Calculate the intersection of two lines.

    Args:
        dict_1 (dict): The first line defined by its slope and intercept.
        dict_2 (dict): The second line defined by its slope and intercept.

    Returns:
        tuple: The (x, y) coordinates of the intersection.
    """
    m_1 = dict_1["slope"]
    b_1 = dict_1["intercept"]

    m_2 = dict_2["slope"]
    b_2 = dict_2["intercept"]

    x = (b_1 - b_2) / (m_2 - m_1)
    y = (b_1 * m_2 - b_2 * m_1) / (m_2 - m_1)

    return x, y


def get_perp_line(
    y_nondim: float, sec1: asb.WingXSec, sec2: asb.WingXSec, xi_c=0.25
) -> Dict:
    """
    Calculate the perpendicular line at a given nondimensional spanwise position.

    Args:
        y_nondim (float): Nondimensional spanwise position.
        sec1 (asb.WingXSec): The root wing section.
        sec2 (asb.WingXSec): The tip wing section.
        xi_c (float, optional): Nondimensional chord position. Defaults to 0.25.

    Returns:
        dict: Information about the perpendicular line and chord.
    """
    x_root, y_root, _ = get_xyz_le_from_xsec(sec1)
    x_tip, y_tip, _ = get_xyz_le_from_xsec(sec2)

    span = y_tip - y_root
    y_le = y_nondim * span
    x_le = get_spanwise_equal_chord_line(x_root, y_root, x_tip, y_tip, inverse=True)(
        y_le
    )

    lambda_qc = calculate_mean_sweep_angle(sec1, sec2, xi_c, radians=True)
    dict_perp_qc = {
        "slope": -np.tan(lambda_qc),
        "intercept": -np.tan(lambda_qc) * -x_le + y_le,
    }

    x_te_root = x_root + sec1.chord
    x_te_tip = x_tip + sec2.chord

    dict_te = get_spanwise_equal_chord_line(
        x_te_root, y_root, x_te_tip, y_tip, return_as_dict=True
    )

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


def get_interpolation_weights(
    y_nondim: float,
    x_nondims: Union[List, np.ndarray],
    sec1: asb.WingXSec,
    sec2: asb.WingXSec,
    xi_c: float = 0.25,
) -> Tuple[np.ndarray]:
    """
    Calculate interpolation weights for root and tip airfoils.

    Args:
        y_nondim (float): Nondimensional spanwise position.
        x_nondims (array): Array of nondimensional chord positions.
        sec1 (asb.WingXSec): The root wing section.
        sec2 (asb.WingXSec): The tip wing section.
        xi_c (float, optional): Nondimensional chord position. Defaults to 0.25.

    Returns:
        tuple: Arrays of interpolation weights for root and tip airfoils.
    """
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


def interpolate_airfoil(af: asb.Airfoil, kind: str = "cubic") -> Tuple[Callable]:
    """
    Interpolate airfoil coordinates using a specified method.

    Args:
        af (asb.Airfoil): The airfoil to interpolate.
        kind (str, optional): The interpolation method. Defaults to "cubic".

    Returns:
        tuple: Interpolated top and bottom surface functions.
    """
    top_function = interp1d(
        af.upper_coordinates()[:, 0],
        af.upper_coordinates()[:, 1],
        kind=kind,
        fill_value="extrapolate",
    )

    bottom_function = interp1d(
        af.lower_coordinates()[:, 0],
        af.lower_coordinates()[:, 1],
        kind=kind,
        fill_value="extrapolate",
    )

    return top_function, bottom_function


def interpolate_airfoil_chordwise(af: asb.Airfoil, x_nondims: Union[List, np.ndarray]):
    """
    Interpolate airfoil coordinates along the chordwise direction.

    Args:
        af (asb.Airfoil): The airfoil to interpolate.
        x_nondims (array): Array of nondimensional chord positions.

    Returns:
        array: Interpolated airfoil coordinates.
    """
    top_function, bottom_function = interpolate_airfoil(af)

    af_interpolated_z = np.concatenate(
        (top_function(x_nondims), bottom_function(x_nondims)[::-1])
    )

    return af_interpolated_z


def _ensure_descending_order(arr: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
    """
    Check if the input array or list is in descending order.
    If it is, return the original array or list.
    If it is not, return an inverted copy of the array or list.

    Args:
        arr (Union[list, np.ndarray]): An array or list of floats expected to contain numbers between 0 and 1.

    Returns:
        Union[list, np.ndarray]: The original array or list if it's in descending order,
                                 otherwise an inverted copy of the array or list.
    """
    # Convert list to numpy array if it's not already
    if isinstance(arr, list):
        arr = np.array(arr)

    # Check if the array is in descending order
    if np.all(arr[:-1] >= arr[1:]):
        return arr if isinstance(arr, np.ndarray) else arr.tolist()
    else:
        # Return an inverted copy
        return arr[::-1] if isinstance(arr, np.ndarray) else arr[::-1].tolist()


def find_interpolated_airfoil(
    y_nondim: float,
    x_nondims: Union[List, np.ndarray],
    sec1: asb.WingXSec,
    sec2: asb.WingXSec,
    xi_c: float = 0.25,
):
    """
    Find the interpolated airfoil at a given spanwise position.

    Args:
        y_nondim (float): Nondimensional spanwise position.
        x_nondims (array): Array of nondimensional chord positions.
        sec1 (asb.WingXSec): The root wing section.
        sec2 (asb.WingXSec): The tip wing section.
        xi_c (float, optional): Nondimensional chord position. Defaults to 0.25.

    Returns:
        asb.Airfoil: The interpolated airfoil.
    """

    x_nondims = _ensure_descending_order(x_nondims)

    weights_root, weights_tip = get_interpolation_weights(
        y_nondim, x_nondims, sec1, sec2, xi_c=xi_c
    )

    weights_root_extended = np.concatenate((weights_root, weights_root[::-1]))
    weights_tip_extended = np.concatenate((weights_tip, weights_tip[::-1]))

    af_1 = sec1.airfoil
    af_2 = sec2.airfoil

    af_1_interpolated_z = interpolate_airfoil_chordwise(af_1, x_nondims)
    af_2_interpolated_z = interpolate_airfoil_chordwise(af_2, x_nondims)

    new_af_z = (
        af_1_interpolated_z * weights_root_extended
        + af_2_interpolated_z * weights_tip_extended
    )
    new_af_x = np.concatenate((x_nondims, x_nondims[::-1]))

    coordinates = np.column_stack((new_af_x, new_af_z))

    new_airfoil = asb.Airfoil(coordinates=coordinates)

    return new_airfoil
