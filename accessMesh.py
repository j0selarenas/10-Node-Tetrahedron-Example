from tet10 import Tet10
from numpy import zeros
from scipy.sparse import csr_matrix

def accessMesh(model, mesh, entities, nodeTags, physicalTags, properties, Nelements, DOFS=3):
	fixed_tag, load_tag, body_tag = physicalTags["fixed"], physicalTags["load"], physicalTags["body"]
	K, F = zeros((DOFS*len(nodeTags), DOFS*len(nodeTags))), zeros(DOFS*len(nodeTags))
	Restricted_DOF, Loaded_DOF, volumen = [], [], 0
	conections, XYZ = zeros((Nelements, 10)), {}

	for ent in entities:
		dim, tag = ent[0], ent[1]
		_, elemTags, elemNodeTags = mesh.getElements(dim, tag)
		physicalTags = model.getPhysicalGroupsForEntity(dim, tag)

		if len(elemTags) == 0:
			continue

		if physicalTags[0] == fixed_tag:
			nodos = elemNodeTags[0]
			for ni in nodos:
				ni = 3*(int(ni)-1)
				Restricted_DOF.append([ni, ni+1, ni+2])

		if physicalTags[0] == load_tag:
			nodos = elemNodeTags[0]
			for ni in nodos:
				ni = 3*(int(ni)-1)
				Loaded_DOF.append(ni+1)

		if physicalTags[0] == body_tag:			
			for e in range(len(elemTags[0])):
				eleTag = elemTags[0][e]
				_, nodes, _, _ = mesh.getElement(eleTag)
				xyz = zeros((len(nodes), DOFS))
				for i, n in enumerate(nodes):
					coord, _, _, _ = mesh.get_node(n)
					xyz[i] = coord
					XYZ[3*(int(n)-1)] = coord

				n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = [3*(int(ni)-1) for ni in nodes]

				ke, fe, vol = Tet10(xyz, properties)
				volumen += vol

				conections[int(eleTag-1), :] = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]

				d = [n1, n1+1, n1+2, n2, n2+1, n2+2, n3, n3+1, n3+2, n4, n4+1, n4+2, n5, n5+1, n5+2,
					 n6, n6+1, n6+2, n7, n7+1, n7+2, n8, n8+1, n8+2, n9, n9+1, n9+2, n10, n10+1, n10+2]
				
				for i in range(len(d)):
					p = d[i]
					for j in range(len(d)):
						q = d[j]
						K[p, q] += ke[i, j]
					F[p] += fe[i]

	return(csr_matrix(K), F, Loaded_DOF, Restricted_DOF, conections, XYZ, volumen)