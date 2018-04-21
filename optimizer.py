import numpy as np
from math import sin


def calculate_xy(hand_angles, hand_lengths):
    hand_angles.append(hand_angles[2] * 2.0 / 3.0)
    hand_angles = np.radians(hand_angles)
    cum_angles = np.cumsum(hand_angles)
    cos_angles = np.cos(cum_angles)
    sin_angles = np.sin(cum_angles)
    x_lengths = np.multiply(cos_angles, hand_lengths)
    y_lengths = np.multiply(sin_angles, hand_lengths)
    #print("x lengths ",  x_lengths)
    #print("y lengths ",  y_lengths)
    x = np.sum(x_lengths)
    y = np.sum(y_lengths)
    return [x, y]


def main():
    hand_angles = [-35, 40, 50]
    hand_lengths = [105, 56, 34, 25]
    res = calculate_xy(hand_angles, hand_lengths)
    print("Final", res)

    pass

if __name__ == "__main__":
    main()