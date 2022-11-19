import gmsh
import numpy as np
import opensees as ops
import matplotlib.pyplot as plt
from tet10 import Tet10

file_name = "cantileverBeam20tet.msh"

f = 0 		# N
g = 9.8 	# m/s2
ρ = 1 		# kg/m3
E = 1000 		# N/m2
ν = 0.25
patternTag, ts_tag, matTag = 1, 1, 1

properties = {"E" : E,
			  "nu": ν,
			  "bx": 0,
			  "by": -ρ*g,
			  "bz": 0}

gmsh.initialize()
gmsh.open(file_name)

model = gmsh.model
mesh = model.mesh
entities = model.getEntities()
nodeTags, _, _ = mesh.getNodes(-1, -1)
_, eleTags, _ = mesh.getElements(-1, -1)
Nelements = mesh.getMaxElementTag()

pg1, pg2, pg3 = model.getPhysicalGroups(dim=1), model.getPhysicalGroups(dim=2), model.getPhysicalGroups(dim=3)
fixed_tag, load_tag, body_tag = pg2[0][1], pg1[0][1], pg3[0][1]
Kpy, Fpy = np.zeros((3*len(nodeTags), 3*len(nodeTags))), np.zeros(3*len(nodeTags))
addedNodes = {}

ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 3)
ops.timeSeries("Constant", ts_tag, "-factor", 1.)
ops.pattern("Plain", patternTag, ts_tag, "-fact", 1.)
ops.nDMaterial("ElasticIsotropic", matTag, E, ν, ρ)

for ent in entities:
	dim, tag = ent[0], ent[1]
	_, elemTags, _ = mesh.getElements(dim, tag)
	physicalTags = model.getPhysicalGroupsForEntity(dim, tag)

	if len(elemTags) == 0:
		continue

	if physicalTags[0] == body_tag:
		countnodes = 1			
		for e in range(len(elemTags[0])):
			eleTag = elemTags[0][e]
			_, nodes, _, _ = mesh.getElement(eleTag)
			xyz = np.zeros((len(nodes), 3))
			nodes_cpp = []
			for i, n in enumerate(nodes):
				coord, _, _, _ = mesh.get_node(n)
				if list(coord) not in addedNodes.values():
					addedNodes[countnodes] = list(coord)
					ops.node(countnodes, coord[0], coord[1], coord[2])
					countnodes += 1
				nodes_cpp.append(list(addedNodes.keys())[list(addedNodes.values()).index(list(coord))])
				xyz[i] = coord
 			
			ops.element("TenNodeTetrahedron", int(e+1), *nodes_cpp, matTag)
			
			kpy, fpy, vol = Tet10(xyz, properties)
			n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = [3*(int(ni)-1) for ni in nodes_cpp]

			d = [n1, n1+1, n1+2, n2, n2+1, n2+2, n3, n3+1, n3+2, n4, n4+1, n4+2, n5, n5+1, n5+2,
				 n6, n6+1, n6+2, n7, n7+1, n7+2, n8, n8+1, n8+2, n9, n9+1, n9+2, n10, n10+1, n10+2]
			
			for i in range(len(d)):
				p = d[i]
				for j in range(len(d)):
					q = d[j]
					Kpy[p, q] += kpy[i, j]
				Fpy[p] += fpy[i]

ops.algorithm("Linear")
ops.numberer("Plain")
ops.constraints("Plain")
ops.system("FullGeneral")
ops.integrator("GimmeMCK", 0., 0., 1.)
ops.analysis("Transient")
ops.analyze(1, 0.)

N  = ops.systemSize()
Kcpp = ops.printA("-ret")
Kcpp = np.array(Kcpp)
Kcpp.shape = (N, N)

print("MAX DIFF BETWEEN Kpy & Kcpp:")
print(f"{np.max(Kpy - Kcpp) = }\n")
print("MIN DIFF BETWEEN Kpy & Kcpp:")
print(f"{np.min(Kpy - Kcpp) = }\n")
print("MAX VALUES FOR Kpy & Kcpp:")
print(f"{np.max(Kpy) = } {np.max(Kcpp) = }\n")
print("MIN VALUES FOR Kpy & Kcpp:")
print(f"{np.min(Kpy) = } {np.min(Kcpp) = }\n")

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.title(f"Element {eleTag}")
ax1.set_title("K python")
ax1.matshow(Kpy)
ax2.set_title("K cpp")
ax2.matshow(Kcpp)
plt.show()