import numpy as np
import scipy as sp
from scipy.spatial.distance import sqeuclidean
from scipy.optimize import minimize, basinhopping
from math import fabs, acos, atan2, pi

switch_size = 19.05
switch_finger_angle = 20


def calculate_xy(switch_pos, switch_angle, finger_angle, hand_lengths):
    angles = np.asarray((finger_angle, finger_angle * 2.0 / 3.0, 180-switch_finger_angle, -switch_angle))
    angles = angles
    angles = np.flipud(angles)
    angles = np.radians(angles)
    cum_angles = np.cumsum(angles)
    cos_angles = np.cos(cum_angles)
    sin_angles = np.sin(cum_angles)
    lengths = np.append(hand_lengths[1:], switch_size * 0.5)
    lengths = np.flipud(lengths)
    x_lengths = np.multiply(cos_angles, lengths)
    y_lengths = np.multiply(sin_angles, lengths)
    lengths = np.column_stack((x_lengths, y_lengths))
    positions = switch_pos + np.cumsum(lengths, axis=0)
    proximal_vec = positions[3] - positions[2]
    palm_vec = np.array((0, 0)) - positions[3]

    proximal_vec_norm = proximal_vec / hand_lengths[1]
    palm_vec_norm = palm_vec / np.linalg.norm(palm_vec)

    d = proximal_vec_norm.dot(palm_vec_norm)

    proximal_palm_angle = acos(np.clip(-proximal_vec_norm.dot(palm_vec_norm), -1, 1)) - pi

    if proximal_palm_angle < 0:
        proximal_palm_angle = 0
        palm_vec_norm = np.array((cos_angles[3], sin_angles[3]))

    palm_angle = np.degrees(pi*0.5-atan2(-palm_vec_norm[0], -palm_vec_norm[1]))

    return positions[3] + palm_vec_norm * hand_lengths[0], np.asarray((-palm_angle, np.degrees(proximal_palm_angle), finger_angle))


def get_switch_cost(switch_pos, switch_angle, hand_lengths):
    def f(angle):
        xy, _ = calculate_xy(switch_pos, switch_angle, angle[0], hand_lengths)
        return np.sum(np.abs(xy))

    minimizer = dict(method="L-BFGS-B", bounds=((0, 80),), tol=1e-10)
    min_res = basinhopping(f, 40, niter=10, minimizer_kwargs=minimizer)
    #print(min_res)

    _, angles = calculate_xy(switch_pos, switch_angle, min_res.x[0], hand_lengths)

    return fabs(min_res.fun) * 100 + fabs(angles[0])


def find_angles(switch_pos, switch_angle, hand_lengths):
    def f(angle):
        xy, _ = calculate_xy(switch_pos, switch_angle, angle[0], hand_lengths)
        return np.sum(np.abs(xy))

    minimizer = dict(method="L-BFGS-B", bounds=((0, 80),), tol=1e-10)
    min_res = basinhopping(f, 40, minimizer_kwargs=minimizer)
    print(min_res)

    _, angles = calculate_xy(switch_pos, switch_angle, min_res.x[0], hand_lengths)
    final_angle = angles.tolist()
    print("RESULT=%s;" % str(final_angle))


def main():
    hand_lengths = [105, 56, 34, 25]
    switch_pos = (200, 0)

    def f(angle):
        return get_switch_cost(switch_pos, angle[0], hand_lengths)

    minimizer = dict(method="L-BFGS-B", bounds=((0,90),), tol=1e-10)
    min_res = basinhopping(f, 45, niter=10, minimizer_kwargs=minimizer)
    #min_res = minimize(f, 45 , method="L-BFGS-B", bounds=[(0,90)], tol=1e-10)
    print (min_res)

    get_switch_cost(switch_pos, min_res.x[0], hand_lengths)
    find_angles(switch_pos, min_res.x[0], hand_lengths)


def old_cod():
    hand_lengths = [105, 56, 34, 25]
    switch_pos = (175, 20)
    switch_angle = 70

    def f(angles):
        xy, angle = calculate_xy(switch_pos, switch_angle, angles, hand_lengths)
        return np.sum(np.abs(xy))

    minimizer = dict(method="L-BFGS-B", bounds=((0, 80),(0,80)), tol=1e-10)
    min_res = basinhopping(f, [40, 40], minimizer_kwargs=minimizer)
    print(min_res)

    _, palm_angle = calculate_xy(switch_pos, switch_angle, min_res.x, hand_lengths)
    final_angle = np.concatenate(([palm_angle], min_res.x)).tolist()
    print("RESULT=%s;" % str(final_angle))




if __name__ == "__main__":
    main()