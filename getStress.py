from tet10 import Tet10
from numpy import array, zeros

def stress(conections, U, XYZ, properties, Nelements, globalEleTags):
	σx, σy, σz = zeros((Nelements, 10)), zeros((Nelements, 10)), zeros((Nelements, 10))
	σxy, σyz, σzx = zeros((Nelements, 10)), zeros((Nelements, 10)), zeros((Nelements, 10))
	εx, εy, εz = zeros((Nelements, 10)), zeros((Nelements, 10)), zeros((Nelements, 10))
	εxy, εyz, εzx = zeros((Nelements, 10)), zeros((Nelements, 10)), zeros((Nelements, 10))

	for eleTag in globalEleTags:
		n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = conections[int(eleTag-1)]

		ue = array([U[int(n1)], U[int(n1+1)], U[int(n1+2)], U[int(n2)], U[int(n2+1)], U[int(n2+2)],
					U[int(n3)], U[int(n3+1)], U[int(n3+2)], U[int(n4)], U[int(n4+1)], U[int(n4+2)],
					U[int(n5)], U[int(n5+1)], U[int(n5+2)], U[int(n6)], U[int(n6+1)], U[int(n6+2)],
					U[int(n7)], U[int(n7+1)], U[int(n7+2)], U[int(n8)], U[int(n8+1)], U[int(n8+2)],
					U[int(n9)], U[int(n9+1)], U[int(n9+2)], U[int(n10)], U[int(n10+1)], U[int(n10+2)]])

		xyz = array([XYZ[int(n1)], XYZ[int(n2)], XYZ[int(n3)], XYZ[int(n4)], XYZ[int(n5)],
					 XYZ[int(n6)], XYZ[int(n7)], XYZ[int(n8)], XYZ[int(n9)], XYZ[int(n10)]])

		εe, σe = Tet10(xyz, properties, ue=ue)

		εx[int(eleTag-1)]  = εe[:, 0]
		εy[int(eleTag-1)]  = εe[:, 1]
		εz[int(eleTag-1)]  = εe[:, 2]
		εxy[int(eleTag-1)] = 0.5*εe[:, 3]
		εyz[int(eleTag-1)] = 0.5*εe[:, 4]
		εzx[int(eleTag-1)] = 0.5*εe[:, 5]

		σx[int(eleTag-1)]  = σe[:, 0]
		σy[int(eleTag-1)]  = σe[:, 1]
		σz[int(eleTag-1)]  = σe[:, 2]
		σxy[int(eleTag-1)] = σe[:, 3]
		σyz[int(eleTag-1)] = σe[:, 4]
		σzx[int(eleTag-1)] = σe[:, 5]

	return(εx, εy, εz, εxy, εyz, εzx, σx, σy, σz, σxy, σyz, σzx)