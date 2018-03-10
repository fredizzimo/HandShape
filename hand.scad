PINKY_FINGER = [[45, 13, 12],[27, 12, 9], [23, 13, 8.5]];
RING_FINGER = [[58, 15, 16],[36.5, 14, 11], [27.5, 13.5, 9.5]];
MIDDLE_FINGER = [[62, 16, 17],[38.3, 15, 14], [26.5, 13.5, 11]];
INDEX_FINGER = [[56, 18, 16],[34, 15, 13.5], [25, 13.5, 10.5]];

PINKY_POS = [77, -38.5, 0];
RING_POS = [91, -20, 0];
MIDDLE_POS = [99, 0, 0];
INDEX_POS = [105, 21.5, 0];

PINKY_ANGLES = [10, 20, 2/3 * 20];
RING_ANGLES = [20, 30, 2/3 * 30];
MIDDLE_ANGLES = [30, 40, 2/3 * 40];
INDEX_ANGLES = [40, 50, 2/3 * 50];


module FingerSegment(length, width, thickness)
{
	cube([length, width, thickness]);
}

module Finger(sizes, angles)
{
	s = sizes;
	a = angles;
	rotate(a = a[0], v = [0, 1, 0])
	translate([s[0][0], 0, (s[0][2] - s[1][2]) / 2])
	{
		rotate(a = a[1], v = [0, 1, 0])
		translate([s[1][0], 0, (s[1][2] - s[2][2]) / 2]) 
		{
			// Distals
			rotate(a = a[2], v = [0, 1, 0])
			translate([0, -s[2][1] / 2, -s[2][2] / 2]) 
			color("red") FingerSegment(s[2][0], s[2][1], s[2][2]);
		}
		//Intermediates
		rotate(a = a[1], v = [0, 1, 0])
		translate([0, -s[1][1] / 2, -s[1][2] / 2]) 
		color("green")FingerSegment(s[1][0], s[1][1], s[1][2]);
	}
	// Proximates
	rotate(a = a[0], v = [0, 1, 0])
	translate([0, -s[0][1] / 2, -s[0][2] / 2]) 
	color("blue") FingerSegment(s[0][0], s[0][1], s[0][2]);
}

module Palm(size)
{
	translate([PINKY_POS.x - size.x, -size.y + INDEX_POS.y + INDEX_FINGER[0][1] / 2, -size.z / 2])
	cube(size);
}

translate(PINKY_POS) Finger(PINKY_FINGER, PINKY_ANGLES);
translate(RING_POS) Finger(RING_FINGER, RING_ANGLES);
translate(MIDDLE_POS) Finger(MIDDLE_FINGER, MIDDLE_ANGLES);
translate(INDEX_POS) Finger(INDEX_FINGER, INDEX_ANGLES);
Palm([58, 80.35, INDEX_FINGER[0][2]]);
