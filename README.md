# 10-node Tetrahedron Example

Note that you should only run "main.py", this code will go through the following codes (in the same order they'll be mentioned):

	1. accessMesh.py
	2. getDisplacements.py
	3. getStress.py

Codes 1 & 3 use the Tet10 function (function thats within tet10.py). This function can get the stiffness, body forces, volume, stress and strains from a ten-node tetrahedron element.

Thanks to this function, it'll now be possible to design all kinds of shapes with a ten-node tetrahedron, and predict it's behaviour in a very precise way!

Huge thanks to Carlos Felippa for publishing the "Advanced Finite Element Methods", since tet10.py was based on said book.
