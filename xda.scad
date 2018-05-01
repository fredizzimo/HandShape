SWITCH_HEIGHT = 9;
CYLINDRICAL_RADIUS = 35;
CYLINDRICAL_DEPTH = 0.6;
SWITCH_BOTTOM_WIDTH = 18;
SWITCH_TOP_WIDTH = 14.5;
SWITCH_CORNER_RADIUS = 1.5;

module SwitchCutout()
{
	translate([-10, 0, CYLINDRICAL_RADIUS + SWITCH_HEIGHT - CYLINDRICAL_DEPTH])
	rotate([0,90,0])
	cylinder(r=CYLINDRICAL_RADIUS, h=20, $fn=100);
}

module Switch()
{

	difference()
	{	
		hull()
		{
			cube([SWITCH_BOTTOM_WIDTH,SWITCH_BOTTOM_WIDTH,0.00001], center=true);

			height = 0.01;
			$fn=50;	
			translate([0,0,SWITCH_HEIGHT-height / 2])
			{
				minkowski()
				{
					cube([SWITCH_TOP_WIDTH-SWITCH_CORNER_RADIUS, SWITCH_TOP_WIDTH-SWITCH_CORNER_RADIUS, height], center=true);
					translate([0, 0, -height])
					cylinder(r=SWITCH_CORNER_RADIUS, h=height);
				}
			}
		}
		SwitchCutout();
		rotate([0, 0, 90])
		SwitchCutout();
	}
}

Switch();
