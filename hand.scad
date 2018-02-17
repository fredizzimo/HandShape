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

Finger([[55, 18, 17],[34, 15, 13.5], [25, 13.5, 10.5]]);
