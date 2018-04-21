module SideCutout()
{
	translate([0, -14 / 2, -1.5])
	rotate(90, [0, 1, 0])
	linear_extrude(height = 14, center = true)
	polygon([[0, 0], [3, 0], [3, 0.5]]);
}

module SwitchCutout(depth, scale)
{
	scale([scale, scale, 1.0])
	difference()
	{
		translate([-14 / 2, -14 / 2, -1.5 - 3])
		cube([14, 14, depth / 2 + 1.5 + 3]);

		SideCutout();

		mirror([0, 1, 0])
		SideCutout();
	}

	cube([14, 13, depth], center = true);

	translate([-16 / 2, -5 / 2, -depth / 2])
	cube([16, 5, depth / 2 - 1.5]);
}

module SwitchTest()
{
	difference()
	{
		translate([0, 0, -4 / 2])
		cube([20, 20, 4], center = true);
		
		SwitchCutout(10, 1.01);
	}
}

SwitchTest();
