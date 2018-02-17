PINKY_FINGER = [[45, 13, 12],[27, 12, 9], [23, 13, 8.5]];
RING_FINGER = [[58, 15, 16],[36.5, 14, 11], [27.5, 13.5, 9.5]];
MIDDLE_FINGER = [[62, 16, 17],[38.3, 15, 14], [26.5, 13.5, 11]];
INDEX_FINGER = [[56, 18, 16],[34, 15, 13.5], [25, 13.5, 10.5]];

PINKY_POS = [0, 0, 0];
RING_POS = [0, 20, 0];
MIDDLE_POS = [0, 40, 0];
INDEX_POS = [0, 60, 0];

module FingerSegment(length, width, thickness)
{
	cube([length, width, thickness]);
}

module Finger(configuration)
{
	c = configuration;
	translate([c[0][0], 0, (c[0][2] - c[1][2]) / 2])
	{
		translate([c[1][0], 0, (c[1][2] - c[2][2]) / 2]) 
		{
			translate([0, -c[2][1] / 2, -c[2][2] / 2]) 
			color("red") FingerSegment(c[2][0], c[2][1], c[2][2]);
		}
		translate([0, -c[1][1] / 2, -c[1][2] / 2]) 
		color("green")FingerSegment(c[1][0], c[1][1], c[1][2]);
	}
	translate([0, -c[0][1] / 2, -c[0][2] / 2]) 
	color("blue") FingerSegment(c[0][0], c[0][1], c[0][2]);
}

translate(PINKY_POS) Finger(PINKY_FINGER);
translate(RING_POS) Finger(RING_FINGER);
translate(MIDDLE_POS) Finger(MIDDLE_FINGER);
translate(INDEX_POS) Finger(INDEX_FINGER);
