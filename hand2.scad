//=====================================
// This is public Domain Code
// Contributed by: Brandon Bowman
// 12 November 2014 => 4 December 2014
//=====================================



 /*use anatomic naming convention
distalPhalanx   =>21 (fingertip)
middlePhalanx   =>31
proximalPhalanx =>48 (finger base ...closest to palm)

(numbers above are weighted averages of total finger length)

thumb
indexFinger
middleFinger
ringFinger
pinkyFinger
*/

///////////////////////////////////begin Global variables//////////////////////////////

taper=2.5;
distalRatio= .21;
middleRatio= .31;
proximalRatio= .48;
indexPalm=2;
middlePalm=2;
ringPalm=2;
pinkPalm=2;

indexLength=5/6;
middleLength =1;
ringLength=3/4;
pinkyLength=2/3;
thumbLength=4/5;

palmLength=100;
handBoneSpacing=20;
handTilt=0;

spreadThumb=0;
tiltThumb=0;
rotateThumb=0;




//////////////////////controls for individual finger curl//////////////////////

thumbCurlProx=10;
thumbCurlmiD=10;
thumbCurlDist=10;

indexCurlProx=20;
indexCurlmiD=10;
indexCurlDist=10;

middleCurlProx=20;
middleCurlmiD=10;
middleCurlDist=10;

ringCurlProx=20;
ringCurlmiD=10;
ringCurlDist=10;

pinkyCurlProx=20;
pinkyCurlmiD=10;
pinkyCurlDist=10;

//handle finger tip extension error


////////////////////////////////////end Global variables///////////////////////////////



////////////////////////////////////begin user input///////////////////////////////
/*
press F5 to compile 
press F6 to compile and render
To fully render the file...it will take ~30 minutes.


*/

//example syntax here
//hand(fingerLength,fingerWidth,knuckleSize,fatHand)


//hand(80,24,17,1,8,60,50,50);
//mirror([0,1,0])
//scale([.5,.5,.5])
handFlatBase(100,25,17,1,10);


/////////////////////////////////////end user input////////////////////////////////

////////////////////////////////////////begin code////////////////////////////////////


module distalPhalanx(length, width,curl){
	sphere(r=width/2,$fn=32);
	rotate([curl,0,0])

	intersection(){
		translate([0,0,width/3])
		scale([1,1,2.6])
		sphere(r=width/2,$fn=32);
	
		fatFinger(length,width);
	}
}


module middlePhalanx(length, width, curl2,curl1){
	rotate([curl2,0,0]){
	sphere(r=(width+taper)/2,$fn=32);

	translate([0,0,length*middleRatio/2])
	cylinder(center=true, h=length*.31, r1=(width+taper)/2, r2=width/2,$fn=32);

	translate([0,0,length*middleRatio])
	distalPhalanx(length,width, curl1);
	}
}


module proximalPhalanx(length, width, curl3,curl2,curl1){
	rotate([curl3,0,0]){
		sphere((width/2)+taper,$fn=32);

		translate([0,0,length*proximalRatio/2])
		cylinder(center=true, h=length*proximalRatio,r1=(width/2)+taper,r2=(width+taper)/2, $fn=32);

		translate([0,0,length*proximalRatio])
		middlePhalanx(length, width, curl2,curl1);
	}
}


module finger(fingerLength, fingerWidth, whichDigit, digitShift,knuckleUp,curlProx,curlmiD,curlDist){
	translate ([handBoneSpacing*digitShift,0,knuckleUp])
	proximalPhalanx(fingerLength*whichDigit,fingerWidth,curlProx,curlmiD,curlDist);

	translate([handBoneSpacing*digitShift,0,-palmLength])
	rotate([0,-digitShift,knuckleUp])
	musculature(fingerWidth,(2.4-(whichDigit*whichDigit/4)),(palmLength/fingerWidth),.8,1);
	
}

module thumb(fingerLength, fingerWidth, digitShift,knuckleUp,curlProx,curlmiD,curlDist){
	rotate([0,0,rotateThumb]){
		rotate([0,30+tiltThumb,0]){
			translate ([handBoneSpacing*digitShift,0,knuckleUp])
				rotate([0,0,280+spreadThumb]){
				proximalPhalanx(fingerLength*thumbLength,fingerWidth*5/4,curlProx/3,curlmiD/3,curlDist/3);
				
			
				//palm musculature [thumb]
				rotate([curlProx/3,0,0])
				musculature(fingerWidth,2.4,(proximalRatio/fingerWidth)*100,.8,1);
			}
		}
	}
}


module fingers(fingerLength, fingerWidth,spread){

	//thumb();
	translate([0,-18,-palmLength*.8])
	rotate([spread*1.6,30,0])
	thumb(fingerLength, fingerWidth, 1.8, 6, thumbCurlProx,thumbCurlmiD,thumbCurlDist);
//	thumb(fingerLength, fingerWidth, 1.8, 6, spreadThumb, thumbCurlProx,thumbCurlmiD,thumbCurlDist);

	
	//index
	rotate([0,spread*1,0])
	finger(fingerLength, fingerWidth, indexLength, 1.8, 6, indexCurlProx,indexCurlmiD,indexCurlDist);
	
	//middle
	rotate([0,spread*.5,0])
	finger(fingerLength, fingerWidth, middleLength, .60, 9, middleCurlProx,middleCurlmiD,middleCurlDist);
	
	//ring
	rotate([0,spread*-.5,0])
	finger(fingerLength, fingerWidth, ringLength, -.60, 3, ringCurlProx,ringCurlmiD,ringCurlDist);
	
	//pinky
	rotate([0,spread*-1,0])
	finger(fingerLength, fingerWidth, pinkyLength, -1.8, 0, pinkyCurlProx,pinkyCurlmiD,pinkyCurlDist);

}


module hand(fingerLength,fingerWidth,knuckleSize,fatHand,spread){
	fingers(fingerLength,fingerWidth,spread);

	difference(){

		union(){
		
			//index
			translate([1.7*handBoneSpacing,0,-7])
			rotate([0,10,0])
			bone(palmLength*.95,fingerWidth*fatHand,3,1.3,1.6,palmLength);
		
			//middle
			translate([.6*handBoneSpacing,0,-0])
			rotate([0,3.5,0])
			bone(palmLength*1,fingerWidth*fatHand,3,1.3,2,palmLength);
		
			//ring
			translate([-.6*handBoneSpacing,0,-6])
			rotate([0,-3.5,0])
			bone(palmLength*.95,fingerWidth*fatHand,3,1.3,2,palmLength);

			//pinky
			translate([-1.7*handBoneSpacing,0,-13])
			rotate([0,-10,0])
			bone(palmLength*.9,fingerWidth*fatHand,3,1.1,1.6,palmLength);


		}
	}
}

module handFlatBase(fingerLength,fingerWidth,knuckleSize,fatHand,spread){

	difference(){
		rotate([-15,handTilt,0])	
		hand(fingerLength,fingerWidth,knuckleSize,fatHand,spread);
		
		//translate([0,0,-(palmLength*.95+40)])
		//#cylinder(h=25,r=150,$fn=6);
		}
}

/////////////////////////////////////begin components/////////////////////////////////

module bone(length,diameter,narrow,widen,lengthen, palmLength){
	translate([0,0,-palmLength/15])
	scale([widen,1,lengthen])
	sphere((diameter/2)+taper);

	translate([0,0,-length*1/6-palmLength/15])
	cylinder(center=true, h=length/3, r1=(diameter+narrow)/2,r2=(diameter/2)+taper, $fn=64);
	translate([0,0,-length*3/6])
	cylinder(center=true, h=length/2.8, r=(diameter+narrow)/2, $fn=64);

	translate([0,0,-length*5/6])
	cylinder(center=true, h=length/3, r2=(diameter+narrow)/2,r1=(diameter/2)+taper, $fn=64);
	translate([0,0,-length*3/3])
	sphere((diameter/2)+taper);
}


module fatFinger(length,width){
	union(){
		translate([0,0,length*distalRatio/1.45])
		cylinder(center=true, h=length*distalRatio*1.7, r=width/2,$fn=32);

		translate([0,0,length*distalRatio/2+(width/2)])
		scale([1,1,1.2])
		sphere(width/2);

	}
}


module musculature(fingerWidth,muscleSize,stretch,squeeze,spread){
	scale([spread,squeeze,1])
	sphere(fingerWidth/2*muscleSize,$fn=32);

	difference(){
		scale([spread,squeeze,stretch])
		sphere(fingerWidth/2*muscleSize,$fn=32);

		translate([0,0,-fingerWidth/2*muscleSize*stretch])
		cube(center=true,[fingerWidth*muscleSize*spread,fingerWidth*muscleSize*squeeze,fingerWidth*muscleSize*stretch]);
	}
}
//////////////////////////////////////end components//////////////////////////////////

/////////////////////////////////////////end code/////////////////////////////////////
