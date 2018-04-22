import numpy as np
import scipy as sp
from scipy.spatial.distance import sqeuclidean
from scipy.optimize import minimize, basinhopping
from math import fabs

switch_size = 19.05
switch_finger_angle = 20

def calculate_xy(switch_pos, switch_angle, hand_angles, hand_lengths):
    angles = np.concatenate((hand_angles, (hand_angles[1] * 2.0 / 3.0, -switch_finger_angle, -switch_angle)))
    angles = -angles
    angles = np.flipud(angles)
    angles = np.radians(angles)
    cum_angles = np.cumsum(angles)
    cos_angles = np.cos(cum_angles)
    sin_angles = np.sin(cum_angles)
    lengths = np.append(hand_lengths, -switch_size * 0.5)
    lengths = np.flipud(lengths)
    x_lengths = np.multiply(cos_angles, lengths)
    y_lengths = np.multiply(sin_angles, lengths)
    x = np.sum(x_lengths)
    y = np.sum(y_lengths)
    xy = np.array((x,y))
    return (switch_pos - xy, np.degrees(cum_angles[4]))

def main():
    hand_lengths = [105, 56, 34, 25]
    switch_pos = (150, 20)
    switch_angle = 70

    def f(angles):
        xy, angle = calculate_xy(switch_pos, switch_angle, angles, hand_lengths)
        return np.sum(np.abs(xy))


    minimizer = dict(method="L-BFGS-B", bounds=((0, 80),(0,80)), tol=1e-10)
    min_res = basinhopping(f, [1, 1], minimizer_kwargs=minimizer)
    print(min_res)

    _, palm_angle = calculate_xy(switch_pos, switch_angle, min_res.x, hand_lengths)
    final_angle = np.concatenate(([palm_angle], min_res.x)).tolist()
    print("RESULT=%s;" % str(final_angle))




if __name__ == "__main__":
    main()