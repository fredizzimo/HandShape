import numpy as np
import scipy as sp
from scipy.spatial.distance import sqeuclidean
from scipy.optimize import minimize, basinhopping
from math import fabs, acos, atan2, pi

switch_size = 19.05
switch_finger_angle = 20


def calculate_finger(switch_pos, switch_angle, finger_angle, hand_lengths):
    angles = np.asarray((finger_angle, finger_angle * 2.0 / 3.0, 180-switch_finger_angle, switch_angle))
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

    proximal_palm_angle = acos(np.clip(-proximal_vec_norm.dot(palm_vec_norm), -1, 1)) - pi

    if proximal_palm_angle < 0:
        proximal_palm_angle = 0
        palm_vec_norm = np.array((cos_angles[3], sin_angles[3]))

    palm_angle = np.degrees(pi*0.5-atan2(-palm_vec_norm[0], -palm_vec_norm[1]))

    palm_pos = positions[3] + palm_vec_norm * hand_lengths[0]
    final_angles = np.asarray((-palm_angle, np.degrees(proximal_palm_angle), finger_angle))

    effort = np.sum(np.abs(palm_pos)) * 100 + fabs(angles[0])

    return effort, final_angles


def optimize_finger(switch_pos, switch_angle, hand_lengths):
    def f(angle):
        return calculate_finger(switch_pos, switch_angle, angle[0], hand_lengths)[0]

    minimizer = dict(method="L-BFGS-B", bounds=((0, 80),), tol=1e-10)
    min_res = basinhopping(f, 40, niter=10, minimizer_kwargs=minimizer)

    return min_res.x[0]


def optimize_switch_angle(switch_pos, hand_lengths):
    def f(switch_angle):
        finger_angle = optimize_finger(switch_pos, switch_angle[0], hand_lengths)
        return calculate_finger(switch_pos, switch_angle[0], finger_angle, hand_lengths)[0]

    minimizer = dict(method="L-BFGS-B", bounds=((-80, 80),), tol=1e-10)
    min_res = basinhopping(f, 0, niter=10, minimizer_kwargs=minimizer)

    return min_res.x[0]


def main():
    hand_lengths = [105, 56, 34, 25]
    switch_pos = (200, 0)

    switch_angle = optimize_switch_angle(switch_pos, hand_lengths)
    finger_angle = optimize_finger(switch_pos, switch_angle, hand_lengths)
    _, angles = calculate_finger(switch_pos, switch_angle, finger_angle, hand_lengths)
    final_angle = angles.tolist()
    print("Switch angle:%f" % switch_angle)
    print("RESULT=%s;" % str(final_angle))


if __name__ == "__main__":
    main()