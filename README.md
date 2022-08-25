# 10-node Tetrahedron Example

The following example takes into acount a cantilever beam of dimensions 10x1x1 [m]. The material of said beam is Steel ASTM A36. Finally, the beam is just subjected to it's self weight. The displacements obtained from the simulation can be seen in the following figure:

![image](https://user-images.githubusercontent.com/111939223/186578661-80db98e0-23ed-41cd-b7c1-b28fb156239f.png)

The stress (considering only the X component) was computed as:

![image](https://user-images.githubusercontent.com/111939223/186579516-a2d8ec24-13ff-4aec-bff3-0b8a04471759.png)

Note that you should only run "main.py", this code will go through the following codes (in the same order they'll be mentioned):

	1. accessMesh.py
	2. getDisplacements.py
	3. getStress.py

Codes 1 & 3 use the Tet10 function (function thats within tet10.py). This function can get the stiffness, body forces, volume, stress and strains from a ten-node tetrahedron element.

Thanks to this function, it'll now be possible to design all kinds of shapes with a ten-node tetrahedron, and predict it's behaviour in a very precise way!

Huge thanks to Carlos Felippa for publishing the "Advanced Finite Element Methods", since tet10.py was based on said book.
