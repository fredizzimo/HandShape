import numpy as np
import scipy as sp
from scipy.spatial.distance import sqeuclidean
from scipy.optimize import minimize, basinhopping
from math import fabs

switch_size = 19.05
switch_angle = 0
switch_angle_rad = np.radians(switch_angle)
switch_angle_cos = np.cos(switch_angle_rad)
switch_angle_sin = np.sin(switch_angle_rad)
switch_rot_matrix = np.array(((switch_angle_cos, -switch_angle_sin), (switch_angle_sin, switch_angle_cos)))


def calculate_xy(switch_pos, hand_angles, hand_lengths):
    angles = np.append(hand_angles, [hand_angles[2] * 2.0 / 3.0])
    angles = np.radians(angles)
    cum_angles = np.cumsum(angles)
    cos_angles = np.cos(cum_angles)
    sin_angles = np.sin(cum_angles)
    x_lengths = np.multiply(cos_angles, hand_lengths)
    y_lengths = np.multiply(sin_angles, hand_lengths)
    #print("x lengths ",  x_lengths)
    #print("y lengths ",  y_lengths)
    x = np.sum(x_lengths)
    y = np.sum(y_lengths)
    xy = -np.array((x, y))
    xy = switch_rot_matrix.dot(xy)
    switch_direction = np.array((switch_size / 2.0, 0))
    return np.array(switch_pos) + switch_direction + xy


def main():
    hand_angles = [-49.94985602,  28.39865415,  61.26461685]
    hand_lengths = [105, 56, 34, 25]
    switch_pos = (150, 0)
    res = calculate_xy(switch_pos, hand_angles, hand_lengths)
    print("Final", res)

    def f(angles):
        xy = calculate_xy(switch_pos, angles, hand_lengths)
        return np.sum(np.abs(xy))


    print(f(hand_angles))

    #min_res = minimize(f, [0, 1, 1], method="TNC", bounds=((-50, 50),(0, 80),(0,80)), tol=1e-10, options={"maxiter": 500})
    #min_res = minimize(f, [0, 1, 1], method="L-BFGS-B", bounds=((-50, 50),(0, 80),(0,80)), tol=1e-10)
    minimizer = dict(method="L-BFGS-B", bounds=((-50, 50),(0, 80),(0,80)), tol=1e-10)
    min_res = basinhopping(f, [0, 1, 1], minimizer_kwargs=minimizer)
    print(min_res)

    print(calculate_xy(switch_pos, min_res.x, hand_lengths))




if __name__ == "__main__":
    main()