PINKY_FINGER = [[45, 13, 12],[27, 12, 9], [23, 13, 8.5]];
RING_FINGER = [[58, 15, 16],[36.5, 14, 11], [27.5, 13.5, 9.5]];
MIDDLE_FINGER = [[62, 16, 17],[38.3, 15, 14], [26.5, 13.5, 11]];
INDEX_FINGER = [[56, 18, 16],[34, 15, 13.5], [25, 13.5, 10.5]];

PINKY_POS = [77, -38.5, 0];
RING_POS = [91, -20, 0];
MIDDLE_POS = [99, 0, 0];
INDEX_POS = [105, 21.5, 0];

PINKY_ANGLES = [10, 20, 2/3 * 20, -20];
RING_ANGLES = [20, 30, 2/3 * 30, -10];
MIDDLE_ANGLES = [30, 40, 2/3 * 40, 0];
INDEX_ANGLES = [40, 50, 2/3 * 50, 20];

FINGER_BONE_POS = 2/3; // The vertical positions of the joints

WRIST_WIDTH = 58;
PALM_LENGTH = 61;
OUTSIDE_KNUCKLE_THICKNESS = 23;
INSIDE_KNUCKLE_THICKNESS = 28;
PALM_BASE_THICKNESS = 41;


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

module TranslateAndBendTwo(prev, cur, angle)
{
	translate([0, (prev.y - cur.y) / 2, (prev.z - cur.z)])
	TranslateAndBendOne(cur, angle, 0)
	children();
}

module TranslateAndBendOne(cur, angle, side_angle)
{
	translate([0, 0, cur.z * FINGER_BONE_POS])
	rotate(a = angle, v = Y_AXIS)
	rotate(a = side_angle, v = Z_AXIS)
	translate([0, 0, -cur.z * FINGER_BONE_POS])
	children();
}

module Finger(sizes, angles)
{
	proximate_s = sizes[PROXIMATE];
	intermediate_s = sizes[INTERMEDIATE];
	distal_s = sizes[DISTAL];

	proximate_a = angles[PROXIMATE];
	intermediate_a = angles[INTERMEDIATE];
	distal_a = angles[DISTAL];
	side_a = angles[3];
	
	translate([0, -proximate_s.y / 2, -proximate_s.z / 2]) 
	TranslateAndBendOne(proximate_s, proximate_a, side_a)
	{
		translate([proximate_s.x, 0, 0])
		{
			color("green")
			hull()
			{
				cube([0.01, proximate_s.y, proximate_s.z]);
				TranslateAndBendTwo(proximate_s, intermediate_s, intermediate_a)
				{
					//Intermediates
					cube(intermediate_s);
				}
			}

			TranslateAndBendTwo(proximate_s, intermediate_s, intermediate_a)
			{

				translate([intermediate_s.x, 0, 0])
				color("red")
				hull()
				{
					cube([0.01, intermediate_s.y, intermediate_s.z]);
					TranslateAndBendTwo(intermediate_s, distal_s, distal_a)
					{
						// Distals
						cube(distal_s);
					}
				}
			}
		 }
		// Proximates
		color("blue") cube(proximate_s);
	}
}

function CylinderAngle(h, r1, r2) = 
	let(y=abs(r1-r2))
	atan(y / h);

module Palm()
{
	hull()
	{
		outside = [PALM_LENGTH, PALM_BASE_THICKNESS / 2, OUTSIDE_KNUCKLE_THICKNESS / 2];
		outside_cylinder_angle = CylinderAngle(outside[0], outside[1], outside[2]);
		outside_pos = abs(PINKY_POS.y - outside[2]);
		wrist_half_width = WRIST_WIDTH / 2;
		diff = outside_pos - wrist_half_width;
		outside_angle = atan(diff / outside[0]);
		
		translate([PINKY_POS.x, PINKY_POS.y, -outside[2]])
		rotate(a = -outside_cylinder_angle - outside_angle, v = Z_AXIS)
		translate(-[outside[0], 0, 0])
		rotate(a = 90 - outside_cylinder_angle, v = Y_AXIS)
		cylinder(outside[0], outside[1], outside[2]);

		inside = [PALM_LENGTH + INDEX_POS.x - PINKY_POS.x, PALM_BASE_THICKNESS / 2, INSIDE_KNUCKLE_THICKNESS / 2];
		inside_cylinder_angle = CylinderAngle(inside[0], inside[1], inside[2]);

		translate([INDEX_POS.x, INDEX_POS.y, -inside[2]])
		rotate(a = inside_cylinder_angle, v = Z_AXIS)
		translate(-[inside[0], 0, 0])
		rotate(a = 90 - inside_cylinder_angle, v = Y_AXIS)
		cylinder(inside[0], inside[1], inside[2]);

	}
}

translate(PINKY_POS) Finger(PINKY_FINGER, PINKY_ANGLES);
translate(RING_POS) Finger(RING_FINGER, RING_ANGLES);
translate(MIDDLE_POS) Finger(MIDDLE_FINGER, MIDDLE_ANGLES);
translate(INDEX_POS) Finger(INDEX_FINGER, INDEX_ANGLES);
Palm();
