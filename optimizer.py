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

    effort = np.sum(np.abs(palm_pos)) * 100 + fabs(np.degrees(angles[0]))

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


def optimize_switch(switch_pos, hand_lengths):
    def f(angles):
        return calculate_finger(switch_pos, angles[0], angles[1], hand_lengths)[0]

    minimizer = dict(method="L-BFGS-B", bounds=((-80, 80), (0, 80)), tol=1e-10)
    min_res = basinhopping(f, (0, 0), niter=10, minimizer_kwargs=minimizer)

    return min_res.x[0], min_res.x[1]


def get_switch_positions(switch_pos, angles):
    switch_angles = np.array(angles)
    switch_angles = np.radians(switch_angles)
    cos_angles = np.cos(switch_angles)
    sin_angles = np.sin(switch_angles)
    switch_lengths = np.full(len(angles), switch_size)
    x_lengths = np.multiply(cos_angles, switch_lengths)
    y_lengths = np.multiply(sin_angles, switch_lengths)
    lengths = np.column_stack((x_lengths, y_lengths))
    positions = switch_pos + np.cumsum(lengths, axis=0)
    prev_positions = np.roll(positions, 1, axis=0)
    prev_positions[0] = switch_pos
    mid_positions = prev_positions + (positions - prev_positions) * 0.5
    return mid_positions


def optimize_switches(hand_lengths, num):
    def f(params):
        switch_angles = params[:num]
        finger_angles = params[num:-2]
        switch_pos = params[-2:]
        positions = get_switch_positions(switch_pos, switch_angles)
        r = 0
        for sp, sa, fa in zip(positions, params[:num], finger_angles):
            r += calculate_finger(sp, sa, fa, hand_lengths)[0]

        return r

    bs = (-80, 80)
    bf = (0, 80)
    bx = (hand_lengths[0] - hand_lengths[3], np.sum(hand_lengths) + switch_size)
    by = (-np.sum(hand_lengths[1:]), hand_lengths[0])

    initial_switch_angle = 0
    initial_finger_angle = 0
    initial_switch_pos = [hand_lengths[0], 0]

    bounds = np.concatenate((
        np.full((num, 2), bs),
        np.full((num, 2), bf),
        (bx,),
        (by,)))

    initial_values = np.concatenate((
        np.full(num, initial_switch_angle),
        np.full(num, initial_finger_angle),
        (initial_switch_pos[0],),
        (initial_switch_pos[1],)))

    minimizer = dict(method="L-BFGS-B", bounds=bounds, tol=1e-10)
    min_res = basinhopping(f, initial_values, niter=100, minimizer_kwargs=minimizer)
    switch_angles = min_res.x[:num]
    finger_angles = min_res.x[num:-2]
    switch_pos = min_res.x[-2:]
    switch_positions = get_switch_positions(switch_pos, switch_angles)

    print(min_res)

    for sp, sa, fa in zip(switch_positions, switch_angles, finger_angles):
        _, angles = calculate_finger(sp, sa, fa, hand_lengths)
        final_angle = angles.tolist()
        print("Switch position:%s" % str(sp))
        print("Switch angle:%f" % sa)
        print("RESULT=%s;" % str(final_angle))



def main():
    hand_lengths = [105, 56, 34, 25]
    switch_pos = (170, 0)

    optimize_switches( hand_lengths, 3)

    switch_pos = (179.40929736, 1.48011764)

    switch_angle, finger_angle = optimize_switch(switch_pos, hand_lengths)
    _, angles = calculate_finger(switch_pos, switch_angle, finger_angle, hand_lengths)
    final_angle = angles.tolist()
    print("Switch angle:%f" % switch_angle)
    print("RESULT=%s;" % str(final_angle))


if __name__ == "__main__":
    main()