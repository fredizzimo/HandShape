import numpy as np
from math import fabs, atan2, pi, pow
from collections import namedtuple
import re
import ast
import itertools
import argparse
import pickle
import contextlib
import jinja2
import json
import pyclipper
import time
import pygmo as pg
import random

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


def get_covering_area(points, depth, direction):
    polygons = []
    for a, b in zip(points[:-1], points[1:]):
        dir = b - a
        dir_norm = dir / np.linalg.norm(dir)
        if direction:
            dir_rot = np.array((-dir_norm[1], dir_norm[0]))
        else:
            dir_rot = np.array((dir_norm[1], dir_norm[0]))
        rot_vector = dir_rot * depth
        ab = np.array((a, b))
        polygon = np.concatenate((ab, np.flipud(ab)+ rot_vector))
        polygons.append(pyclipper.scale_to_clipper(polygon))
    return polygons


def calculate_finger(switch_pos, switch_angle, finger_angle, hand_lengths, forbidden_area):
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

    proximal_angle = pi*0.5-atan2(-proximal_vec_norm[0], -proximal_vec_norm[1])
    palm_angle = pi*0.5-atan2(-palm_vec_norm[0], -palm_vec_norm[1])

    proximal_palm_angle = palm_angle - proximal_angle

    palm_pos = positions[3] + palm_vec_norm * hand_lengths[0]
    final_angles_radians= np.asarray((-palm_angle, proximal_palm_angle, angles[-1], angles[-1] * 2.0 / 3.0))
    final_angles = np.degrees(final_angles_radians)

    #TODO: this should come from the configuration
    finger_width = 15
    finger_area = get_covering_area(np.concatenate(((palm_pos,), np.flipud(positions[1:]))), finger_width, True)

    clipper = pyclipper.Pyclipper()
    clipper.AddPaths(finger_area, pyclipper.PT_SUBJECT, True)
    clipper.AddPaths(forbidden_area, pyclipper.PT_CLIP, True)
    res = clipper.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

    overlapping_area = 0
    for poly in res:
        overlapping_area += 1000

    effort = np.sum(np.abs(palm_pos)) * 100 + fabs(final_angles[0])
    if final_angles[1] < 0:
        effort += 10 * np.abs(final_angles[1])
    elif final_angles[1] > 90:
        effort += 10 * (final_angles[1] - 90)
    else:
        effort += final_angles[1] * 0.1

    effort += final_angles[2] * 0.01
    effort += overlapping_area
    effort /= 100.0

    return effort, final_angles


def calculate_switches(switch_pos, angles):
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

    forbidden_area = get_covering_area(np.concatenate(((switch_pos,), positions)), 25, False)

    return mid_positions, forbidden_area


def optimize_switches(hand_lengths, num_switches, num_passes=2, iter_success=100):
    bs = (-80, 80)
    bf = (0, 80)
    bx = (hand_lengths[0] - hand_lengths[3], np.sum(hand_lengths) + switch_size)
    by = (-np.sum(hand_lengths[1:]), hand_lengths[0])
    bf = (0, 80)

    def scale(a, s):
        return a * (s[1] - s[0]) + s[0]

    class Problem:
        def fitness(self, params):
            switch_angles = scale(params[:num_switches], bs)
            finger_angles = scale(params[num_switches:-2], bf)
            switch_pos = np.array((scale(params[-2], bx), scale(params[-1], by)))
            positions, forbidden_area = calculate_switches(switch_pos, switch_angles)
            r = 0
            for sp, sa, fa in zip(positions, switch_angles, finger_angles):
                r += calculate_finger(sp, sa, fa, hand_lengths, forbidden_area)[0]

            return (r,)

        def get_bounds(self):
            num_params = num_switches * 2 + 2
            return (np.full(num_params, 0), np.full(num_params, 1))

    prob = pg.problem(Problem())
    #0.328703
    #0.32865749
    #0.32484355963
    #0.324843559629
    generation = 0

    batch = 50
    pop_size = 100

    archi = pg.archipelago()
    for variant in range(18):
        algo = pg.algorithm(pg.sade(gen=batch, variant=variant+1, variant_adptv=2, memory=True))
        archi.push_back(algo=algo, prob=prob, size=pop_size)

    old_fevals = np.full(18, 0);

    def get_archi_champion(archi):
        f = np.array(archi.get_champions_f()).flatten()
        s = np.argsort(f)
        x = archi.get_champions_x()[s[0]]
        f = archi.get_champions_f()[s[0]][0]
        return x, f, s[0]

    pg.mp_island.init_pool()
    pg.mp_island.resize_pool(pg.mp_island.get_pool_size() - 1)

    best = np.finfo(np.float64).max
    num_stagnated = 0

    for i in range(500):
        archi.evolve(1)
        archi.wait()
        islands = list(archi)

        best_x_f = []
        x = np.array(archi.get_champions_x())
        f = np.array(archi.get_champions_f())
        dt = np.dtype([("x", x.dtype, (x.shape[1],)), ("f", f.dtype, (f.shape[1],))])
        all_x_f=np.empty(0, dtype=dt)
        for island in islands:
            pop = island.get_population()
            f = pop.champion_f
            x = pop.champion_x
            best_x_f.append((x, f))
            f = pop.get_f()
            x = pop.get_x()
            a = np.empty(len(x), dtype=dt)
            a["x"]=x
            a["f"]=f
            all_x_f = np.concatenate((all_x_f, a))

        generation += batch
        x, f, i = get_archi_champion(archi)
        if np.isclose(f, best, atol=1e-9, rtol=1e-20):
            num_stagnated += 1
            if num_stagnated > 5:
                break
        else:
            best = f
            num_stagnated = 0
        print("Generation %i: variant %i: best %f, same for: %i" % (generation, i, f, num_stagnated))
        print(np.array(best_x_f).T[1])

        _, i = np.unique(np.round(all_x_f["x"], 10), axis=0, return_index=True)
        unique_x_f = np.empty(len(i), dtype=dt)
        unique_x_f["x"] = all_x_f["x"][i]
        unique_x_f["f"] = all_x_f["f"][i]

        for i, island in enumerate(islands):
            pop = island.get_population()
            f = pop.get_f().flatten()
            s = np.argsort(f)
            j = 0
            for fx in best_x_f:
                if fx[1] != pop.champion_f:
                    pop.set_xf(int(s[-j]), fx[0], fx[1])
                    j = j + 1
            island.set_population(pop)

        new_fevals = np.array([i.get_population().problem.get_fevals() for i in islands])
        fevals = new_fevals - old_fevals
        print("Evaluations %s" % (fevals))

        for i, f in enumerate(fevals):
            if f < 3 * pop_size:
                new_island = random.randrange(len(islands))
                print("Stagnated island %i, replacing with %i" %(i, new_island))
                pop = islands[i].get_population()
                new_pop = islands[new_island].get_population()
                xs = new_pop.get_x();
                fs = new_pop.get_f();
                for j, (x, f) in enumerate(zip(xs, fs)):
                    pop.set_xf(j, x, f)
                islands[i].set_population(pop)

        old_fevals = new_fevals

    x = get_archi_champion(archi)[0]

    switch_angles = scale(x[:num_switches], bs)
    finger_angles = scale(x[num_switches:-2], bf)
    switch_pos = np.array((scale(x[-2], bx), scale(x[-1], by)))
    switch_positions, forbidden_area = calculate_switches(switch_pos, switch_angles)

    total_effort = 0
    switches = []
    for sp, sa, fa in zip(switch_positions, switch_angles, finger_angles):
        effort, angles = calculate_finger(sp, sa, fa, hand_lengths, forbidden_area)
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

    switch_to_press = r.switches[2]
    hand = {
        "palm_angle": switch_to_press.finger_angles[0],
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

    hand["index"]["angle"] = get_finger_angles(switch_to_press)

    keys = [[s.switch_position[0], s.switch_position[1], -s.switch_angle] for s in r.switches]

    with open("keyboard.scad", "w") as output_file:
        output_file.write(template.render(
            hand=hand, keys=json.dumps(keys)
        ))


if __name__ == "__main__":
    main()