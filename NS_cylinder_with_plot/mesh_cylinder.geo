// Gmsh project created on Fri Mar 06 13:21:11 2026
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2.2, 0, 0, 1.0};
//+
Point(3) = {2.2, 0.41, 0, 1.0};
//+
Point(4) = {0, 0.41, 0, 1.0};
//+
Point(5) = {0.2, 0.2, 0, 1.0};
//+
Point(6) = {0.15, 0.2, 0, 1.0};
//+
Point(7) = {0.25, 0.2, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Circle(4) = {6, 5, 7};
//+
Circle(5) = {7, 5, 6};
//+
Line(6) = {4, 3};
//+
Curve Loop(1) = {6, -3, -2, -1};
//+
Curve Loop(2) = {4, 5};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve("inlet", 7) = {1};
//+
Physical Curve("outlet", 8) = {3};
//+
Physical Curve("circle", 9) = {4, 5};
//+
Physical Curve("walls", 10) = {6, 2};
//+
Physical Surface("surface", 11) = {1};
