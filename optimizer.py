import numpy as np
import scipy as sp
from scipy.spatial.distance import sqeuclidean
from scipy.optimize import minimize, basinhopping
from math import fabs

switch_size = 19.05
switch_angle = 20
switch_angle_rad = np.radians(switch_angle)
switch_angle_cos = np.cos(switch_angle_rad)
switch_angle_sin = np.sin(switch_angle_rad)
switch_rot_matrix = np.array(((switch_angle_cos, -switch_angle_sin), (switch_angle_sin, switch_angle_cos)))


def calculate_xy(switch_pos, hand_angles, hand_lengths):
    angles = np.concatenate((hand_angles, (hand_angles[1] * 2.0 / 3.0, -switch_angle)))
    angles = -angles
    angles = np.flipud(angles)
    angles = np.radians(angles)
    cum_angles = np.cumsum(angles)
    cos_angles = np.cos(cum_angles)
    sin_angles = np.sin(cum_angles)
    lengths = np.flipud(hand_lengths)
    x_lengths = np.multiply(cos_angles, lengths)
    y_lengths = np.multiply(sin_angles, lengths)
    x = np.sum(x_lengths)
    y = np.sum(y_lengths)
    return (switch_pos + np.array((switch_size / 2.0, 0)) - np.array((x, y)), np.degrees(cum_angles[3]))

def main():
    hand_angles = [33.37951475829356, 3.0914731867019123]
    hand_lengths = [105, 56, 34, 25]
    switch_pos = (200, 0)

    def f(angles):
        xy, angle = calculate_xy(switch_pos, angles, hand_lengths)
        return np.sum(np.abs(xy))


    minimizer = dict(method="L-BFGS-B", bounds=((0, 80),(0,80)), tol=1e-10)
    min_res = basinhopping(f, [1, 1], minimizer_kwargs=minimizer)
    print(min_res)

    _, palm_angle = calculate_xy(switch_pos, min_res.x, hand_lengths)
    final_angle = np.concatenate(([palm_angle], min_res.x)).tolist()
    print("RESULT=%s;" % str(final_angle))




if __name__ == "__main__":
    main()