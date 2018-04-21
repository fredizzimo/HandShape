import numpy as np
import scipy as sp
from scipy.spatial.distance import sqeuclidean
from scipy.optimize import minimize
from math import fabs


def calculate_xy(hand_angles, hand_lengths):
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
    return np.array((x, y))


def main():
    hand_angles = [-35, 40, 50]
    hand_lengths = [105, 56, 34, 25]
    res = calculate_xy(hand_angles, hand_lengths)
    print("Final", res)

    def f(angles):
        xy = calculate_xy(angles, hand_lengths)
        dist = sqeuclidean(res, xy)
        return fabs(angles[0]) * 0.01 + dist
        if (dist > 0.1):
            return dist + 90
        else:
            return fabs(angles[0]) + dist


    print(f(hand_angles))

    #min_res = minimize(f, [0, 1, 1], method="TNC", bounds=((-50, 50),(0, 80),(0,80)), tol=1e-10, options={"maxiter": 500})
    min_res = minimize(f, [0, 1, 1], method="L-BFGS-B", bounds=((-50, 50),(0, 80),(0,80)), tol=1e-10)
    print(min_res)

    print(calculate_xy(min_res.x, hand_lengths))




if __name__ == "__main__":
    main()