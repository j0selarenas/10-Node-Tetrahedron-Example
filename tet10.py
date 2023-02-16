from numpy import array, zeros 	# pip install numpy
from scipy.linalg import det 	# pip install scipy

def Tet10(xyz, properties, ue=None):
	"""
	Function based somewhat on Chapters 16 & 17 of Advanced Finite Element Methods by Carlos Felippa.

	Inputs: xyz - array of nodal coordinates (10x3).
			properties - dictionary of material properties.
			ue - deformed shape (30x1).

	Outputs: ke - stiffness matrix (30x30).
			 fe - force vector (30x1).
			 εe - strain vector (10x1).
			 σe - stress vector (10x1).
	
	Note: To output εe and σe, the deformed shape must be provided. Otherwise, only ke and fe will be output.
	"""

	# Element properties:
	E  = properties["E"]
	ν  = properties["nu"]
	bx = properties["bx"]
	by = properties["by"]
	bz = properties["bz"]
	
	# ------------------------------------------------------------------------------------------------------------------------------------------------------------

	# Stress-strain matrix:
	E_ = E/((1 + ν)*(1 - 2*ν)) * array([[1 - ν, ν, ν, 0, 0, 0],
										[ν, 1 - ν, ν, 0, 0, 0],
										[ν, ν, 1 - ν, 0, 0, 0],
										[0, 0, 0, 1/2 - ν, 0, 0],
										[0, 0, 0, 0, 1/2 - ν, 0],
										[0, 0, 0, 0, 0, 1/2 - ν]])

	# ------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	# Nodal coordinates:
	x1, x2, x3, x4, x5 = xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[4, 0]
	y1, y2, y3, y4, y5 = xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[4, 1]
	z1, z2, z3, z4, z5 = xyz[0, 2], xyz[1, 2], xyz[2, 2], xyz[3, 2], xyz[4, 2]
	
	x6, x7, x8, x9, x10 = xyz[5, 0], xyz[6, 0], xyz[7, 0], xyz[8, 0], xyz[9, 0]
	y6, y7, y8, y9, y10 = xyz[5, 1], xyz[6, 1], xyz[7, 1], xyz[8, 1], xyz[9, 1]
	z6, z7, z8, z9, z10 = xyz[5, 2], xyz[6, 2], xyz[7, 2], xyz[8, 2], xyz[9, 2]

	# ------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	# If there is no given displacement field, Tet10 computed the stiffness matrix and the force vector.
	if ue is None:
		# Gauss rule (wi, ζ1, ζ2, ζ3)
		α, β = 0.58541020, 0.13819660
		Gauss_rule = [(1/4, α, β, β, β),
					  (1/4, β, α, β, β),
					  (1/4, β, β, α, β),
					  (1/4, β, β, β, α)]

		# Variables to hold the element stiffness matrix and the element load vector for each Gauss point:
		ke, fe = zeros((30, 30)), zeros((30, 1))

		# Body force field over the element:
		b = array([[bx], [by], [bz]])

		# Variable to hold the element volume:
		volume = 0

		# --------------------------------------------------------------------------------------------------------------------------------------------------------

		# Loop over the Gauss points:
		for wi, ζ1, ζ2, ζ3, ζ4 in Gauss_rule:
			# The shape functions are given by Equation 17.2 (AFEM):
			# Note that N10 and N9 are modified to account for the fact that the nodes 9 and 10 are in a different position than they should be.
			N1, N2, N3, N4, N5  = ζ1*(2*ζ1-1), ζ2*(2*ζ2-1), ζ3*(2*ζ3-1), ζ4*(2*ζ4-1), 4*ζ1*ζ2
			N6, N7, N8, N10, N9 = 4*ζ2*ζ3, 4*ζ3*ζ1, 4*ζ1*ζ4, 4*ζ2*ζ4, 4*ζ3*ζ4
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------
			
			# Shape function similar to 16.33 (AFEM) due to the node-wise displacement ordering.
			N = array([[N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0, N9, 0, 0, N10, 0, 0],
					   [0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0, N9, 0, 0, N10, 0],
					   [0, 0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0, N9, 0, 0, N10]])

			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The derivatives of the shape functions with respect to the natural coordinates are given by:
			dN1_dζ1, dN2_dζ1, dN3_dζ1, dN4_dζ1, dN5_dζ1  = 4*ζ1 - 1, 0, 0, 0, 4*ζ2
			dN6_dζ1, dN7_dζ1, dN8_dζ1, dN10_dζ1, dN9_dζ1 = 0, 4*ζ3, 4*ζ4, 0, 0

			dN1_dζ2, dN2_dζ2, dN3_dζ2, dN4_dζ2, dN5_dζ2  = 0, 4*ζ2 - 1, 0, 0, 4*ζ1
			dN6_dζ2, dN7_dζ2, dN8_dζ2, dN10_dζ2, dN9_dζ2 = 4*ζ3, 0, 0, 4*ζ4, 0

			dN1_dζ3, dN2_dζ3, dN3_dζ3, dN4_dζ3, dN5_dζ3  = 0, 0, 4*ζ3 - 1, 0, 0
			dN6_dζ3, dN7_dζ3, dN8_dζ3, dN10_dζ3, dN9_dζ3 = 4*ζ2, 4*ζ1, 0, 0, 4*ζ4

			dN1_dζ4, dN2_dζ4, dN3_dζ4, dN4_dζ4, dN5_dζ4  = 0, 0, 0, 4*ζ4 - 1, 0
			dN6_dζ4, dN7_dζ4, dN8_dζ4, dN10_dζ4, dN9_dζ4 = 0, 0, 4*ζ1, 4*ζ2, 4*ζ3

			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The following derivatives are computed to obtain the Jacobian matrix. These equations can be found in Equation 17.9 (AFEM).
			dx_dζ1 = x1*dN1_dζ1 + x2*dN2_dζ1 + x3*dN3_dζ1 + x4*dN4_dζ1 + x5*dN5_dζ1 + x6*dN6_dζ1 + x7*dN7_dζ1 + x8*dN8_dζ1 + x9*dN9_dζ1 + x10*dN10_dζ1
			dy_dζ1 = y1*dN1_dζ1 + y2*dN2_dζ1 + y3*dN3_dζ1 + y4*dN4_dζ1 + y5*dN5_dζ1 + y6*dN6_dζ1 + y7*dN7_dζ1 + y8*dN8_dζ1 + y9*dN9_dζ1 + y10*dN10_dζ1
			dz_dζ1 = z1*dN1_dζ1 + z2*dN2_dζ1 + z3*dN3_dζ1 + z4*dN4_dζ1 + z5*dN5_dζ1 + z6*dN6_dζ1 + z7*dN7_dζ1 + z8*dN8_dζ1 + z9*dN9_dζ1 + z10*dN10_dζ1

			dx_dζ2 = x1*dN1_dζ2 + x2*dN2_dζ2 + x3*dN3_dζ2 + x4*dN4_dζ2 + x5*dN5_dζ2 + x6*dN6_dζ2 + x7*dN7_dζ2 + x8*dN8_dζ2 + x9*dN9_dζ2 + x10*dN10_dζ2
			dy_dζ2 = y1*dN1_dζ2 + y2*dN2_dζ2 + y3*dN3_dζ2 + y4*dN4_dζ2 + y5*dN5_dζ2 + y6*dN6_dζ2 + y7*dN7_dζ2 + y8*dN8_dζ2 + y9*dN9_dζ2 + y10*dN10_dζ2
			dz_dζ2 = z1*dN1_dζ2 + z2*dN2_dζ2 + z3*dN3_dζ2 + z4*dN4_dζ2 + z5*dN5_dζ2 + z6*dN6_dζ2 + z7*dN7_dζ2 + z8*dN8_dζ2 + z9*dN9_dζ2 + z10*dN10_dζ2

			dx_dζ3 = x1*dN1_dζ3 + x2*dN2_dζ3 + x3*dN3_dζ3 + x4*dN4_dζ3 + x5*dN5_dζ3 + x6*dN6_dζ3 + x7*dN7_dζ3 + x8*dN8_dζ3 + x9*dN9_dζ3 + x10*dN10_dζ3
			dy_dζ3 = y1*dN1_dζ3 + y2*dN2_dζ3 + y3*dN3_dζ3 + y4*dN4_dζ3 + y5*dN5_dζ3 + y6*dN6_dζ3 + y7*dN7_dζ3 + y8*dN8_dζ3 + y9*dN9_dζ3 + y10*dN10_dζ3
			dz_dζ3 = z1*dN1_dζ3 + z2*dN2_dζ3 + z3*dN3_dζ3 + z4*dN4_dζ3 + z5*dN5_dζ3 + z6*dN6_dζ3 + z7*dN7_dζ3 + z8*dN8_dζ3 + z9*dN9_dζ3 + z10*dN10_dζ3
			
			dx_dζ4 = x1*dN1_dζ4 + x2*dN2_dζ4 + x3*dN3_dζ4 + x4*dN4_dζ4 + x5*dN5_dζ4 + x6*dN6_dζ4 + x7*dN7_dζ4 + x8*dN8_dζ4 + x9*dN9_dζ4 + x10*dN10_dζ4
			dy_dζ4 = y1*dN1_dζ4 + y2*dN2_dζ4 + y3*dN3_dζ4 + y4*dN4_dζ4 + y5*dN5_dζ4 + y6*dN6_dζ4 + y7*dN7_dζ4 + y8*dN8_dζ4 + y9*dN9_dζ4 + y10*dN10_dζ4
			dz_dζ4 = z1*dN1_dζ4 + z2*dN2_dζ4 + z3*dN3_dζ4 + z4*dN4_dζ4 + z5*dN5_dζ4 + z6*dN6_dζ4 + z7*dN7_dζ4 + z8*dN8_dζ4 + z9*dN9_dζ4 + z10*dN10_dζ4
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The Jacobian matrix can be simplified to a 3x3 matrix by subtracting the first column of J from the last three columns. Check Equation 17.15 (AFEM).
			J = array([[dx_dζ2 - dx_dζ1, dx_dζ3 - dx_dζ1, dx_dζ4 - dx_dζ1],
					   [dy_dζ2 - dy_dζ1, dy_dζ3 - dy_dζ1, dy_dζ4 - dy_dζ1],
					   [dz_dζ2 - dz_dζ1, dz_dζ3 - dz_dζ1, dz_dζ4 - dz_dζ1]])
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The determinant of the Jacobian matrix is the volume of the tetrahedron. Check Equation 17.14 (AFEM).
			det_J = det(J)

			# If the determinant is less or equal to zero, the node numbering is wrong.
			assert det_J > 0, "The node numbering is wrong! Mapping is not invertible!"

			# ----------------------------------------------------------------------------------------------------------------------------------------------------
			
			# Equation 16.7 (AFEM).
			a1, a2 = y2*(z4-z3)-y3*(z4-z2)+y4*(z3-z2), -y1*(z4-z3)+y3*(z4-z1)-y4*(z3-z1)
			a3, a4 = y1*(z4-z2)-y2*(z4-z1)+y4*(z2-z1), -y1*(z3-z2)+y2*(z3-z1)-y3*(z2-z1)

			b1, b2 = -x2*(z4-z3)+x3*(z4-z2)-x4*(z3-z2), x1*(z4-z3)-x3*(z4-z1)+x4*(z3-z1)
			b3, b4 = -x1*(z4-z2)+x2*(z4-z1)-x4*(z2-z1), x1*(z3-z2)-x2*(z3-z1)+x3*(z2-z1)

			c1, c2 = x2*(y4-y3)-x3*(y4-y2)+x4*(y3-y2), -x1*(y4-y3)+x3*(y4-y1)-x4*(y3-y1)
			c3, c4 = x1*(y4-y2)-x2*(y4-y1)+x4*(y2-y1), -x1*(y3-y2)+x2*(y3-y1)-x3*(y2-y1)
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# Assembling a matrix of the derivatives of the shape functions with respect to the natural coordinates.
			dNi_dζj = array([[dN1_dζ1, dN1_dζ2, dN1_dζ3, dN1_dζ4],
							 [dN2_dζ1, dN2_dζ2, dN2_dζ3, dN2_dζ4],
							 [dN3_dζ1, dN3_dζ2, dN3_dζ3, dN3_dζ4],
							 [dN4_dζ1, dN4_dζ2, dN4_dζ3, dN4_dζ4],
							 [dN5_dζ1, dN5_dζ2, dN5_dζ3, dN5_dζ4],
							 [dN6_dζ1, dN6_dζ2, dN6_dζ3, dN6_dζ4],
							 [dN7_dζ1, dN7_dζ2, dN7_dζ3, dN7_dζ4],
							 [dN8_dζ1, dN8_dζ2, dN8_dζ3, dN8_dζ4],
							 [dN9_dζ1, dN9_dζ2, dN9_dζ3, dN9_dζ4],
							 [dN10_dζ1, dN10_dζ2, dN10_dζ3, dN10_dζ4]])

			# ----------------------------------------------------------------------------------------------------------------------------------------------------
			
			# The following values are computed to obtain the Strain-Displacement matrix. Check Equation 17.24 (AFEM).
			qx1, qx2, qx3, qx4, qx5, qx6, qx7, qx8, qx9, qx10 = ((1/det(J)) * (dNi_dζj @ array([[a1], [a2], [a3], [a4]])).T)[0]
			qy1, qy2, qy3, qy4, qy5, qy6, qy7, qy8, qy9, qy10 = ((1/det(J)) * (dNi_dζj @ array([[b1], [b2], [b3], [b4]])).T)[0]
			qz1, qz2, qz3, qz4, qz5, qz6, qz7, qz8, qz9, qz10 = ((1/det(J)) * (dNi_dζj @ array([[c1], [c2], [c3], [c4]])).T)[0]
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# Strain-Displacement matrix. Check Equation 17.23 (AFEM).
			B = array([[qx1, 0, 0, qx2, 0, 0, qx3, 0, 0, qx4, 0, 0, qx5, 0, 0, qx6, 0, 0, qx7, 0, 0, qx8, 0, 0, qx9, 0, 0, qx10, 0, 0],
					   [0, qy1, 0, 0, qy2, 0, 0, qy3, 0, 0, qy4, 0, 0, qy5, 0, 0, qy6, 0, 0, qy7, 0, 0, qy8, 0, 0, qy9, 0, 0, qy10, 0],
					   [0, 0, qz1, 0, 0, qz2, 0, 0, qz3, 0, 0, qz4, 0, 0, qz5, 0, 0, qz6, 0, 0, qz7, 0, 0, qz8, 0, 0, qz9, 0, 0, qz10],
					   [qy1, qx1, 0, qy2, qx2, 0, qy3, qx3, 0, qy4, qx4, 0, qy5, qx5, 0, qy6, qx6, 0, qy7, qx7, 0, qy8, qx8, 0, qy9, qx9, 0, qy10, qx10, 0],
					   [0, qz1, qy1, 0, qz2, qy2, 0, qz3, qy3, 0, qz4, qy4, 0, qz5, qy5, 0, qz6, qy6, 0, qz7, qy7, 0, qz8, qy8, 0, qz9, qy9, 0, qz10, qy10],
					   [qz1, 0, qx1, qz2, 0, qx2, qz3, 0, qx3, qz4, 0, qx4, qz5, 0, qx5, qz6, 0, qx6, qz7, 0, qx7, qz8, 0, qx8, qz9, 0, qx9, qz10, 0, qx10]])
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The volume is obtained by multiplying the determinant of the Jacobian matrix by the weight of the Gaussian Point.
			volume += wi * det_J

			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The stiffness matrix is obtained by using Equation 17.25 (AFEM).
			ke += wi * B.T @ E_ @ B * det_J

			# The force vector is obtained by using Equation 17.27 (AFEM).
			fe += wi * N.T @ b * det_J
		
		# --------------------------------------------------------------------------------------------------------------------------------------------------------

		# Returning stiffness, body forces and volume.
		return (ke, fe, volume)

	# ------------------------------------------------------------------------------------------------------------------------------------------------------------

	else:
		# Lists to hold stresses and strains for each node:
		εe, σe = zeros((10, 6)), zeros((10, 6))

		# Natural coordinates so that we can then evaluate at each corresponding node.
		Natural_coords = [(1, 0, 0, 0), 		# N1 = 1
						  (0, 1, 0, 0),			# N2 = 1
						  (0, 0, 1, 0), 		# N3 = 1
						  (0, 0, 0, 1), 		# N4 = 1
						  (1/2, 1/2, 0, 0), 	# N5 = 1
						  (0, 1/2, 1/2, 0), 	# N6 = 1
						  (1/2, 0, 1/2, 0), 	# N7 = 1
						  (1/2, 0, 0, 1/2), 	# N8 = 1
						  (0, 1/2, 0, 1/2), 	# N10 = 1
						  (0, 0, 1/2, 1/2)] 	# N9 = 1

		# --------------------------------------------------------------------------------------------------------------------------------------------------------

		# Loop over the Natural coordinates:
		for idx, (ζ1, ζ2, ζ3, ζ4) in enumerate(Natural_coords):
			# The shape functions are given by Equation 17.2 (AFEM):
			# Note that N10 and N9 are modified to account for the fact that the nodes 9 and 10 are in a different position than they should be.
			N1, N2, N3, N4, N5  = ζ1*(2*ζ1-1), ζ2*(2*ζ2-1), ζ3*(2*ζ3-1), ζ4*(2*ζ4-1), 4*ζ1*ζ2
			N6, N7, N8, N10, N9 = 4*ζ2*ζ3, 4*ζ3*ζ1, 4*ζ1*ζ4, 4*ζ2*ζ4, 4*ζ3*ζ4
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------
			
			# Shape function similar to 16.33 (AFEM) due to the node-wise displacement ordering.
			N = array([[N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0, N9, 0, 0, N10, 0, 0],
					[0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0, N9, 0, 0, N10, 0],
					[0, 0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0, N9, 0, 0, N10]])

			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The derivatives of the shape functions with respect to the natural coordinates are given by:
			dN1_dζ1, dN2_dζ1, dN3_dζ1, dN4_dζ1, dN5_dζ1  = 4*ζ1 - 1, 0, 0, 0, 4*ζ2
			dN6_dζ1, dN7_dζ1, dN8_dζ1, dN10_dζ1, dN9_dζ1 = 0, 4*ζ3, 4*ζ4, 0, 0

			dN1_dζ2, dN2_dζ2, dN3_dζ2, dN4_dζ2, dN5_dζ2  = 0, 4*ζ2 - 1, 0, 0, 4*ζ1
			dN6_dζ2, dN7_dζ2, dN8_dζ2, dN10_dζ2, dN9_dζ2 = 4*ζ3, 0, 0, 4*ζ4, 0

			dN1_dζ3, dN2_dζ3, dN3_dζ3, dN4_dζ3, dN5_dζ3  = 0, 0, 4*ζ3 - 1, 0, 0
			dN6_dζ3, dN7_dζ3, dN8_dζ3, dN10_dζ3, dN9_dζ3 = 4*ζ2, 4*ζ1, 0, 0, 4*ζ4

			dN1_dζ4, dN2_dζ4, dN3_dζ4, dN4_dζ4, dN5_dζ4  = 0, 0, 0, 4*ζ4 - 1, 0
			dN6_dζ4, dN7_dζ4, dN8_dζ4, dN10_dζ4, dN9_dζ4 = 0, 0, 4*ζ1, 4*ζ2, 4*ζ3

			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The following derivatives are computed to obtain the Jacobian matrix. These equations can be found in Equation 17.9 (AFEM).
			dx_dζ1 = x1*dN1_dζ1 + x2*dN2_dζ1 + x3*dN3_dζ1 + x4*dN4_dζ1 + x5*dN5_dζ1 + x6*dN6_dζ1 + x7*dN7_dζ1 + x8*dN8_dζ1 + x9*dN9_dζ1 + x10*dN10_dζ1
			dy_dζ1 = y1*dN1_dζ1 + y2*dN2_dζ1 + y3*dN3_dζ1 + y4*dN4_dζ1 + y5*dN5_dζ1 + y6*dN6_dζ1 + y7*dN7_dζ1 + y8*dN8_dζ1 + y9*dN9_dζ1 + y10*dN10_dζ1
			dz_dζ1 = z1*dN1_dζ1 + z2*dN2_dζ1 + z3*dN3_dζ1 + z4*dN4_dζ1 + z5*dN5_dζ1 + z6*dN6_dζ1 + z7*dN7_dζ1 + z8*dN8_dζ1 + z9*dN9_dζ1 + z10*dN10_dζ1

			dx_dζ2 = x1*dN1_dζ2 + x2*dN2_dζ2 + x3*dN3_dζ2 + x4*dN4_dζ2 + x5*dN5_dζ2 + x6*dN6_dζ2 + x7*dN7_dζ2 + x8*dN8_dζ2 + x9*dN9_dζ2 + x10*dN10_dζ2
			dy_dζ2 = y1*dN1_dζ2 + y2*dN2_dζ2 + y3*dN3_dζ2 + y4*dN4_dζ2 + y5*dN5_dζ2 + y6*dN6_dζ2 + y7*dN7_dζ2 + y8*dN8_dζ2 + y9*dN9_dζ2 + y10*dN10_dζ2
			dz_dζ2 = z1*dN1_dζ2 + z2*dN2_dζ2 + z3*dN3_dζ2 + z4*dN4_dζ2 + z5*dN5_dζ2 + z6*dN6_dζ2 + z7*dN7_dζ2 + z8*dN8_dζ2 + z9*dN9_dζ2 + z10*dN10_dζ2

			dx_dζ3 = x1*dN1_dζ3 + x2*dN2_dζ3 + x3*dN3_dζ3 + x4*dN4_dζ3 + x5*dN5_dζ3 + x6*dN6_dζ3 + x7*dN7_dζ3 + x8*dN8_dζ3 + x9*dN9_dζ3 + x10*dN10_dζ3
			dy_dζ3 = y1*dN1_dζ3 + y2*dN2_dζ3 + y3*dN3_dζ3 + y4*dN4_dζ3 + y5*dN5_dζ3 + y6*dN6_dζ3 + y7*dN7_dζ3 + y8*dN8_dζ3 + y9*dN9_dζ3 + y10*dN10_dζ3
			dz_dζ3 = z1*dN1_dζ3 + z2*dN2_dζ3 + z3*dN3_dζ3 + z4*dN4_dζ3 + z5*dN5_dζ3 + z6*dN6_dζ3 + z7*dN7_dζ3 + z8*dN8_dζ3 + z9*dN9_dζ3 + z10*dN10_dζ3
			
			dx_dζ4 = x1*dN1_dζ4 + x2*dN2_dζ4 + x3*dN3_dζ4 + x4*dN4_dζ4 + x5*dN5_dζ4 + x6*dN6_dζ4 + x7*dN7_dζ4 + x8*dN8_dζ4 + x9*dN9_dζ4 + x10*dN10_dζ4
			dy_dζ4 = y1*dN1_dζ4 + y2*dN2_dζ4 + y3*dN3_dζ4 + y4*dN4_dζ4 + y5*dN5_dζ4 + y6*dN6_dζ4 + y7*dN7_dζ4 + y8*dN8_dζ4 + y9*dN9_dζ4 + y10*dN10_dζ4
			dz_dζ4 = z1*dN1_dζ4 + z2*dN2_dζ4 + z3*dN3_dζ4 + z4*dN4_dζ4 + z5*dN5_dζ4 + z6*dN6_dζ4 + z7*dN7_dζ4 + z8*dN8_dζ4 + z9*dN9_dζ4 + z10*dN10_dζ4
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The Jacobian matrix can be simplified to a 3x3 matrix by subtracting the first column of J from the last three columns. Check Equation 17.15 (AFEM).
			J = array([[dx_dζ2 - dx_dζ1, dx_dζ3 - dx_dζ1, dx_dζ4 - dx_dζ1],
					   [dy_dζ2 - dy_dζ1, dy_dζ3 - dy_dζ1, dy_dζ4 - dy_dζ1],
					   [dz_dζ2 - dz_dζ1, dz_dζ3 - dz_dζ1, dz_dζ4 - dz_dζ1]])
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# The determinant of the Jacobian matrix is the volume of the tetrahedron. Check Equation 17.14 (AFEM).
			det_J = det(J)

			# If the determinant is less or equal to zero, the node numbering is wrong.
			assert det_J > 0, "The node numbering is wrong! Mapping is not invertible!"

			# ----------------------------------------------------------------------------------------------------------------------------------------------------
			
			# Equation 16.7 (AFEM).
			a1, a2 = y2*(z4-z3)-y3*(z4-z2)+y4*(z3-z2), -y1*(z4-z3)+y3*(z4-z1)-y4*(z3-z1)
			a3, a4 = y1*(z4-z2)-y2*(z4-z1)+y4*(z2-z1), -y1*(z3-z2)+y2*(z3-z1)-y3*(z2-z1)

			b1, b2 = -x2*(z4-z3)+x3*(z4-z2)-x4*(z3-z2), x1*(z4-z3)-x3*(z4-z1)+x4*(z3-z1)
			b3, b4 = -x1*(z4-z2)+x2*(z4-z1)-x4*(z2-z1), x1*(z3-z2)-x2*(z3-z1)+x3*(z2-z1)

			c1, c2 = x2*(y4-y3)-x3*(y4-y2)+x4*(y3-y2), -x1*(y4-y3)+x3*(y4-y1)-x4*(y3-y1)
			c3, c4 = x1*(y4-y2)-x2*(y4-y1)+x4*(y2-y1), -x1*(y3-y2)+x2*(y3-y1)-x3*(y2-y1)
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# Assembling a matrix of the derivatives of the shape functions with respect to the natural coordinates.
			dNi_dζj = array([[dN1_dζ1, dN1_dζ2, dN1_dζ3, dN1_dζ4],
							 [dN2_dζ1, dN2_dζ2, dN2_dζ3, dN2_dζ4],
							 [dN3_dζ1, dN3_dζ2, dN3_dζ3, dN3_dζ4],
							 [dN4_dζ1, dN4_dζ2, dN4_dζ3, dN4_dζ4],
							 [dN5_dζ1, dN5_dζ2, dN5_dζ3, dN5_dζ4],
							 [dN6_dζ1, dN6_dζ2, dN6_dζ3, dN6_dζ4],
							 [dN7_dζ1, dN7_dζ2, dN7_dζ3, dN7_dζ4],
							 [dN8_dζ1, dN8_dζ2, dN8_dζ3, dN8_dζ4],
							 [dN9_dζ1, dN9_dζ2, dN9_dζ3, dN9_dζ4],
							 [dN10_dζ1, dN10_dζ2, dN10_dζ3, dN10_dζ4]])

			# ----------------------------------------------------------------------------------------------------------------------------------------------------
			
			# The following values are computed to obtain the Strain-Displacement matrix. Check Equation 17.24 (AFEM).
			qx1, qx2, qx3, qx4, qx5, qx6, qx7, qx8, qx9, qx10 = ((1/det(J)) * (dNi_dζj @ array([[a1], [a2], [a3], [a4]])).T)[0]
			qy1, qy2, qy3, qy4, qy5, qy6, qy7, qy8, qy9, qy10 = ((1/det(J)) * (dNi_dζj @ array([[b1], [b2], [b3], [b4]])).T)[0]
			qz1, qz2, qz3, qz4, qz5, qz6, qz7, qz8, qz9, qz10 = ((1/det(J)) * (dNi_dζj @ array([[c1], [c2], [c3], [c4]])).T)[0]
			
			# ----------------------------------------------------------------------------------------------------------------------------------------------------

			# Strain-Displacement matrix. Check Equation 17.23 (AFEM).
			B = array([[qx1, 0, 0, qx2, 0, 0, qx3, 0, 0, qx4, 0, 0, qx5, 0, 0, qx6, 0, 0, qx7, 0, 0, qx8, 0, 0, qx9, 0, 0, qx10, 0, 0],
					   [0, qy1, 0, 0, qy2, 0, 0, qy3, 0, 0, qy4, 0, 0, qy5, 0, 0, qy6, 0, 0, qy7, 0, 0, qy8, 0, 0, qy9, 0, 0, qy10, 0],
					   [0, 0, qz1, 0, 0, qz2, 0, 0, qz3, 0, 0, qz4, 0, 0, qz5, 0, 0, qz6, 0, 0, qz7, 0, 0, qz8, 0, 0, qz9, 0, 0, qz10],
					   [qy1, qx1, 0, qy2, qx2, 0, qy3, qx3, 0, qy4, qx4, 0, qy5, qx5, 0, qy6, qx6, 0, qy7, qx7, 0, qy8, qx8, 0, qy9, qx9, 0, qy10, qx10, 0],
					   [0, qz1, qy1, 0, qz2, qy2, 0, qz3, qy3, 0, qz4, qy4, 0, qz5, qy5, 0, qz6, qy6, 0, qz7, qy7, 0, qz8, qy8, 0, qz9, qy9, 0, qz10, qy10],
					   [qz1, 0, qx1, qz2, 0, qx2, qz3, 0, qx3, qz4, 0, qx4, qz5, 0, qx5, qz6, 0, qx6, qz7, 0, qx7, qz8, 0, qx8, qz9, 0, qx9, qz10, 0, qx10]])
			
			# Computing strain:
			ε = B @ ue
			# Computing stress:
			σ = E_ @ ε

			# Saving data to arrays:
			εe[idx, :] = ε
			σe[idx, :] = σ

		# --------------------------------------------------------------------------------------------------------------------------------------------------------
		
		# Returning strain and stress components.
		return (εe, σe)
