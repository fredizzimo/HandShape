PINKY_FINGER = [[45, 13, 12],[27, 12, 9], [23, 13, 8.5]];
RING_FINGER = [[58, 15, 16],[36.5, 14, 11], [27.5, 13.5, 9.5]];
MIDDLE_FINGER = [[62, 16, 17],[38.3, 15, 14], [26.5, 13.5, 11]];
INDEX_FINGER = [[56, 18, 1],[34, 15, 1], [25, 13.5, 1]];

PINKY_POS = [77, -38.5, 0];
RING_POS = [91, -20, 0];
MIDDLE_POS = [99, 0, 0];
INDEX_POS = [105, 21.5, 0];

FINGER_BONE_POS = 1; // The vertical positions of the joints
WRIST_BONE_POS = [2/3, 1]; // horisontal / vertical

PINKY_INTERDIGITAL_FOLD = 15;
RING_INTERDIGITAL_FOLD = 20;
MIDDLE_INTERDIGITAL_FOLD = 23;

WRIST_WIDTH = 58;
PALM_LENGTH = 61;
OUTSIDE_KNUCKLE_THICKNESS = 1;
INSIDE_KNUCKLE_THICKNESS = 1;
PALM_BASE_THICKNESS = 1;
ARM_LENGTH = 50;


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
	
	translate([0, -proximate_s.y / 2, -proximate_s.z]) 
	TranslateAndBendOne(proximate_s, proximate_a, side_a)
	{
		translate([proximate_s.x, 0, 0])
		{
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
		cube(proximate_s);
	}
}

module InterdigitalFold(sizes, angles, fold)
{
	proximate_s = sizes[PROXIMATE];
	proximate_a = angles[PROXIMATE];
	side_a = angles[3];

	translate([0, -proximate_s.y / 2, -proximate_s.z]) 
	TranslateAndBendOne(proximate_s, proximate_a, side_a)
	{
		{
			translate([0, proximate_s.y, 0])
			rotate(90, X_AXIS)
			linear_extrude(height=proximate_s.y)
			polygon([[0, 0], [fold, 0], [0, proximate_s.z]]);
		}
	}
}

module Fingers(pinky_angles, ring_angles, middle_angles, index_angles)
{
	translate(PINKY_POS) Finger(PINKY_FINGER, pinky_angles);
	translate(RING_POS) Finger(RING_FINGER, ring_angles);
	translate(MIDDLE_POS) Finger(MIDDLE_FINGER, middle_angles);
	translate(INDEX_POS) Finger(INDEX_FINGER, index_angles);
}

module InterdigitalFolds(pinky_angles, ring_angles, middle_angles, index_angles)
{
	hull()
	{
		translate(PINKY_POS) InterdigitalFold(PINKY_FINGER, pinky_angles, PINKY_INTERDIGITAL_FOLD);
		translate(RING_POS) InterdigitalFold(RING_FINGER, ring_angles, PINKY_INTERDIGITAL_FOLD);
	}
	hull()
	{
		translate(RING_POS) InterdigitalFold(RING_FINGER, ring_angles, RING_INTERDIGITAL_FOLD);
		translate(MIDDLE_POS) InterdigitalFold(MIDDLE_FINGER, middle_angles, RING_INTERDIGITAL_FOLD);
	}
	hull()
	{
		translate(MIDDLE_POS) InterdigitalFold(MIDDLE_FINGER, middle_angles, MIDDLE_INTERDIGITAL_FOLD);
		translate(INDEX_POS) InterdigitalFold(INDEX_FINGER, index_angles, MIDDLE_INTERDIGITAL_FOLD);
	}
}

function CylinderAngle(h, r1, r2) = 
	let(y=abs(r1-r2))
	atan(y / h);

module WristSide(length)
{
	wrist_cylinder_radius = PALM_BASE_THICKNESS / 2;
	translate([0, WRIST_WIDTH / 2 - wrist_cylinder_radius, - wrist_cylinder_radius + PINKY_POS.z])
	rotate(a = 90, v = Y_AXIS)
	cylinder(length, r1 = wrist_cylinder_radius, r2 = wrist_cylinder_radius);
}

module Palm()
{
	wrist_length = PINKY_POS.x - PALM_LENGTH;
	hull()
	{
		outside = [PALM_LENGTH, PALM_BASE_THICKNESS / 2, OUTSIDE_KNUCKLE_THICKNESS / 2];
		outside_cylinder_angle = CylinderAngle(outside[0], outside[1], outside[2]);
		outside_pos = abs(PINKY_POS.y - outside[2]);
		wrist_half_width = WRIST_WIDTH / 2;
		diff = outside_pos - wrist_half_width;
		outside_angle = atan(diff / outside[0]);
		
		translate([PINKY_POS.x, PINKY_POS.y, -outside[1]])
		rotate(a = -outside_cylinder_angle - outside_angle, v = Z_AXIS)
		translate(-[outside[0], 0, 0])
		rotate(a = 90 - outside_cylinder_angle, v = Y_AXIS)
		cylinder(outside[0], outside[1], outside[2]);

		inside = [PALM_LENGTH + INDEX_POS.x - PINKY_POS.x, PALM_BASE_THICKNESS / 2, INSIDE_KNUCKLE_THICKNESS / 2];
		inside_cylinder_angle = CylinderAngle(inside[0], inside[1], inside[2]);

		translate([INDEX_POS.x, INDEX_POS.y, -inside[1]])
		rotate(a = inside_cylinder_angle, v = Z_AXIS)
		translate(-[inside[0], 0, 0])
		rotate(a = 90 - inside_cylinder_angle, v = Y_AXIS)
		cylinder(inside[0], inside[1], inside[2]);

		overlap = 1;
		translate([wrist_length - overlap, 0 , 0])
		WristSide(overlap);
		
		translate([wrist_length - overlap, 0 , 0])
		mirror(Y_AXIS)
		WristSide(overlap);
	}
}

module Wrist()
{
	wrist_length = PINKY_POS.x - PALM_LENGTH;
	hull()
	{
		WristSide(wrist_length);
		
		mirror(Y_AXIS)
		WristSide(wrist_length);
	}
}

module Arm(length)
{
	hull()
	{
		mirror(X_AXIS)
		WristSide(length);

		mirror(Y_AXIS)
		mirror(X_AXIS)
		WristSide(length);
	}
}

module RotateWrist(extension, deviation)
{
	half_wrist = WRIST_WIDTH / 2;
	bone_pos = [0, -WRIST_WIDTH * WRIST_BONE_POS.x + half_wrist, PALM_BASE_THICKNESS * WRIST_BONE_POS.y];
	translate(-bone_pos)
	rotate(extension, Y_AXIS)
	rotate(deviation, Z_AXIS)
	translate(bone_pos)
	children();
}

module Hand(extension, deviation, pinky_angles, ring_angles, middle_angles, index_angles)
{
	RotateWrist(extension, deviation)
	{
		Fingers(pinky_angles, ring_angles, middle_angles, index_angles);
		hull()
		{
			InterdigitalFolds();
			Palm();
		}
	}

	hull()
	{
		RotateWrist(extension, deviation)
		{
			Wrist();
		}
		Arm(10);
	}

	Arm(ARM_LENGTH);
}

PINKY_ANGLES = [10, 20, 2/3 * 20, -20];
RING_ANGLES = [20, 30, 2/3 * 30, -10];
MIDDLE_ANGLES = [30, 40, 2/3 * 40, 0];
INDEX_ANGLES = [40, 50, 2/3 * 50, 20];

color("BurlyWood");
Hand(-30, -20, PINKY_ANGLES, RING_ANGLES, MIDDLE_ANGLES, INDEX_ANGLES);
