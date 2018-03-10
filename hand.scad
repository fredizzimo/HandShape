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

FINGER_BONE_POS = 2/3; // The vertical positions of the joints

PROXIMATE = 0;
INTERMEDIATE = 1;
DISTAL = 2;

X_AXIS = [1, 0, 0];
Y_AXIS = [0, 1, 0];
Z_AXIS = [0, 0, 1];


module FingerSegment(length, width, thickness)
{
	cube([length, width, thickness]);
}

module Finger(sizes, angles)
{
	proximate_s = sizes[PROXIMATE];
	intermediate_s = sizes[INTERMEDIATE];
	distal_s = sizes[DISTAL];

	proximate_a = angles[PROXIMATE];
	intermediate_a = angles[INTERMEDIATE];
	distal_a = angles[DISTAL];
	
	translate([0, -proximate_s.y / 2, -proximate_s.z / 2]) 
	rotate(a = proximate_a, v = Y_AXIS)
	{
		translate([proximate_s.x, (proximate_s.y - intermediate_s.y) / 2, (proximate_s.z - intermediate_s.z)])
		rotate(a = intermediate_a, v = Y_AXIS)
		{
			translate([intermediate_s.x, (intermediate_s.y - distal_s.y) / 2, (intermediate_s.z - distal_s.z)])
			rotate(a = distal_a, v = Y_AXIS)
			{
				// Distals
				color("red") FingerSegment(distal_s.x, distal_s.y, distal_s.z);
			}
			//Intermediates
			color("green")FingerSegment(intermediate_s.x, intermediate_s.y, intermediate_s.z);
		 }
		// Proximates
		color("blue") FingerSegment(proximate_s.x, proximate_s.y, proximate_s.z);
	}
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
