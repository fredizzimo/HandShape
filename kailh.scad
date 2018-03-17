module SideCutout()
{
	translate([0, -14 / 2, -1.5])
	rotate(90, [0, 1, 0])
	linear_extrude(height = 14, center = true)
	polygon([[0, 0], [3, 0], [3, 0.5]]);
}

module SwitchCutout(depth)
{
	difference()
	{
		translate([-14 / 2, -14 / 2, -1.5 - 3])
		cube([14, 14, depth / 2 + 1.5 + 3]);

		SideCutout();

		mirror([0, 1, 0])
		SideCutout();
	}

	cube([14, 13, depth], center = true);

	translate([-15.6 / 2, -5 / 2, -depth / 2])
	cube([15.6, 5, depth / 2 - 1.5]);
}

module SwitchTest()
{
	difference()
	{
		translate([0, 0, -6 / 2])
		cube([20, 20, 6], center = true);
		
		SwitchCutout(15);
	}
}

SwitchTest();
