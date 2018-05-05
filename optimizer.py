import numpy as np
from scipy.optimize import basinhopping
from math import fabs, acos, atan2, pi
from collections import namedtuple
import re
import ast
import itertools
import argparse
import pickle
import contextlib
import jinja2
import json

switch_size = 19.05
switch_finger_angle = 20


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class TakeStep(object):
    def __init__(self, stepsize, num_iterations, random_state):
        self._initial_stepsize = stepsize
        self._num_iterations = num_iterations
        self._stepsize = stepsize
        self._random_state = random_state
        self._nstep = 0
        self._accepted_stepsizes = []

    def __call__(self, x):
        self._nstep += 1
        stepsize = self._initial_stepsize / self._num_iterations
        if self._stepsize - stepsize > stepsize:
            self._stepsize -= stepsize
        if self._nstep % 5 == 0:
            print("Current stepsize %f" %self._stepsize)
        x += self._random_state.uniform(-self._stepsize, self._stepsize, np.shape(x))
        return x

    def report(self, accept, f_new, x_new, f_old, x_old):
        if accept:
            self._accepted_stepsizes.append(self._stepsize)
            self._initial_stepsize = np.average(self._accepted_stepsizes[-3:])
            self._stepsize = self._initial_stepsize

OptimizationResultSwitch = namedtuple(
    "OptimizationResultSwitch",
    [
        "switch_position",
        "switch_angle",
        "effort",
        "finger_angles"])

OptimizationResult = namedtuple(
    "OptimizationResult",
    [
        "switches",
        "total_effort"])


def calculate_finger(switch_pos, switch_angle, finger_angle, hand_lengths):
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

    proximal_angle = np.degrees(pi*0.5-atan2(-proximal_vec_norm[0], -proximal_vec_norm[1]))
    palm_angle = np.degrees(pi*0.5-atan2(-palm_vec_norm[0], -palm_vec_norm[1]))

    proximal_palm_angle = palm_angle - proximal_angle

    palm_pos = positions[3] + palm_vec_norm * hand_lengths[0]
    final_angles = np.asarray((-palm_angle, proximal_palm_angle, finger_angle))

    effort = np.sum(np.abs(palm_pos)) * 1000 + fabs(palm_angle) * 10 + fabs(proximal_palm_angle)
    effort /= 1000.0

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


def optimize_switches(hand_lengths, num_switches, num_passes=2, iter_success=100):
    bs = (-80, 80)
    bf = (0, 80)
    bx = (hand_lengths[0] - hand_lengths[3], np.sum(hand_lengths) + switch_size)
    by = (-np.sum(hand_lengths[1:]), hand_lengths[0])
    bf = (0, 80)

    def scale(a, s):
        return a * (s[1] - s[0]) + s[0]

    def f(params):
        switch_angles = scale(params[:num_switches], bs)
        finger_angles = scale(params[num_switches:-2], bf)
        switch_pos = np.array((scale(params[-2], bx), scale(params[-1], by)))
        positions = get_switch_positions(switch_pos, switch_angles)
        r = 0
        for sp, sa, fa in zip(positions, switch_angles, finger_angles):
            r += calculate_finger(sp, sa, fa, hand_lengths)[0]
        return r

    bounds = np.full((num_switches * 2 + 2, 2), (0.0, 1.0))
    initial_values = np.full(num_switches * 2 + 2, 0.5)

    rnd = np.random.RandomState()

    min_res = None
    for _ in range(num_passes):
        take_step = TakeStep(0.5, iter_success, rnd)
        minimizer = dict(method="SLSQP", bounds=bounds, tol=1e-9)
        res = basinhopping(
            f, initial_values, T=0.0000000001, take_step=take_step, niter=10000, niter_success=iter_success,
            minimizer_kwargs=minimizer, seed=rnd, disp=True)
        if min_res is None or res.fun < min_res.fun:
            print("New global minimum")
            print(res)
            min_res = res
    switch_angles = scale(min_res.x[:num_switches], bs)
    finger_angles = scale(min_res.x[num_switches:-2], bf)
    switch_pos = np.array((scale(min_res.x[-2], bx), scale(min_res.x[-1], by)))
    switch_positions = get_switch_positions(switch_pos, switch_angles)

    print(min_res)

    total_effort = 0
    switches = []
    for sp, sa, fa in zip(switch_positions, switch_angles, finger_angles):
        effort, angles = calculate_finger(sp, sa, fa, hand_lengths)
        switches.append(OptimizationResultSwitch(sp, sa, effort, angles))
        total_effort += effort

    return OptimizationResult(switches, total_effort)


def evaluate_openscad_variables(filename, variables):
    output = dict()
    with open(filename) as f:
        regexps = [re.compile(v + " *= *(.*);") for v in variables]
        for l in f.readlines():
            for r, v in zip(regexps, variables):
                m = r.match(l)
                if m:
                    output[v] = ast.literal_eval(m.group(1))
    for v in variables:
        if v not in output:
            raise Exception("Variable %s not defined in %s" % (v, filename))
    return output


def load_variables(finger_names):
    return evaluate_openscad_variables(
        "hand.scad",
        list(itertools.chain.from_iterable(((v + "_FINGER", v + "_POS") for v in finger_names))))


def print_result(result):
    with printoptions(suppress=True, precision=3):
        for switch in result.switches:
            print("Switch position:%s" % switch.switch_position)
            print("Switch angle:%f" % switch.switch_angle)
            print("Switch effort:%f" % switch.effort)
            print("RESULT=%s;" % switch.finger_angles)
        print("Total Effort: %f" % result.total_effort)

def get_finger_angles(switch_result):
    return [switch_result.finger_angles[1], switch_result.finger_angles[2], switch_result.finger_angles[2] * 2.0 / 3.0]

def main():
    parser = argparse.ArgumentParser(description="Optimize HandShape keyboard switch placement")
    parser.add_argument(
        "--npasses",
        default=2,
        type=int,
        help="Number of optimization passes")
    parser.add_argument(
        "--niter",
        default=100,
        type=int,
        help="Consider the result optimal when it has been stable for this many iterations")
    parser.add_argument(
        "--fingers",
        nargs="*",
        default=["PINKY", "RING", "MIDDLE", "INDEX"],
        help="The fingers to optimize (a list of [PINKY, RING, MIDDLLE, and/or INDEX)")
    parser.add_argument(
        "--save",
        type=argparse.FileType("wb"),
        help="Save the optimization result to a file, can be loaded later to skip the optimization step")
    parser.add_argument(
        "--load",
        type=argparse.FileType("rb"),
        help="Load the optimization result from a file, this skips the optimization step")
    args = parser.parse_args()

    if args.load:
        optimization_results = pickle.load(args.load)
        finger_names = list(optimization_results.keys())
        variables = load_variables(finger_names)
    else:
        finger_names = args.fingers
        variables = load_variables(finger_names)

        optimization_results = dict()

        for finger_name in finger_names:
            print("Calculating %s" % finger_name)
            finger_dimensions = np.array(variables["%s_FINGER" % finger_name])
            finger_pos = np.array(variables["%s_POS" % finger_name])
            lengths = np.concatenate((finger_pos[0:1], finger_dimensions[:,0]))
            result = optimize_switches(lengths, 3, args.npasses, args.niter)
            optimization_results[finger_name] = result

        if args.save:
            pickle.dump(optimization_results, args.save)

    for finger_name in finger_names:
        print("Result for %s" % finger_name)
        print_result(optimization_results[finger_name])


    current_finger = "INDEX"
    r = optimization_results[current_finger]

    finger_dimensions = np.array(variables["%s_FINGER" % current_finger])
    finger_pos = np.array(variables["%s_POS" % current_finger])
    lengths = np.concatenate((finger_pos[0:1], finger_dimensions[:,0]))

    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("keyboard.template")

    hand = {
        "palm_angle": r.switches[0].finger_angles[0],
        "pinky": {
            "angle": np.full(3, 0),
        },
        "ring": {
            "angle": np.full(3, 0),
        },
        "middle": {
            "angle": np.full(3, 0),
        },
        "index": {
            "angle": np.full(3, 0),
        }
    }

    temp = r.switches[1]
    calculate_finger(temp.switch_position, temp.switch_angle, temp.finger_angles[2], lengths)
    hand["index"]["angle"] = get_finger_angles(r.switches[1])

    keys = [[s.switch_position[0], s.switch_position[1], -s.switch_angle] for s in r.switches]

    with open("keyboard.scad", "w") as output_file:
        output_file.write(template.render(
            hand=hand, keys=json.dumps(keys)
        ))


if __name__ == "__main__":
    main()