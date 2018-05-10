use <xda.scad>


module KeyRow(switches)
{
	for(switch = switches)
	{
		pos = [switch[0], 0, switch[1]];
		rotation = [0, switch[2], 0];
		translate(pos)
		rotate(rotation)
		translate([0, 0, -9])
		Switch();
	}
}

KeyRow([[0, 0, 10], [20, 0, 20], [40, 0, 30]]);
