import gmsh
from numpy import arange
from getStress import stress
from accessMesh import accessMesh
from getDisplacements import deformedShape

def main():
	gmsh.initialize()
	gmsh.open(file_name)

	model = gmsh.model
	mesh = model.mesh
	entities = model.getEntities()
	nodeTags, _, _ = mesh.getNodes(-1, -1)
	_, eleTags, _ = mesh.getElements(-1, -1)
	globalEleTags = eleTags[2]
	Nelements = mesh.getMaxElementTag()

	pg1, pg2, pg3 = model.getPhysicalGroups(dim=1), model.getPhysicalGroups(dim=2), model.getPhysicalGroups(dim=3)
	physicalTags = {"fixed": pg2[0][1], "load": pg1[0][1], "body": pg3[0][1]}

	K, F, Loaded_DOF, Restricted_DOF, conections, XYZ, volume = accessMesh(model, mesh, entities, nodeTags, physicalTags, properties, Nelements)
	print("DONE COMPUTING STIFFNESS MATRIX!")

	U = deformedShape(Restricted_DOF, Loaded_DOF, nodeTags, F, K, f)
	print("DONE COMPUTING DEFORMATIONS!")

	εx, εy, εz, εxy, εyz, εzx, σx, σy, σz, σxy, σyz, σzx = stress(conections, U, XYZ, properties, Nelements, globalEleTags)
	print("DONE COMPUTING STRESSES!")

	print(f"Volume = {round(volume, 5)}")

	gmsh.view.addHomogeneousModelData(tag=gmsh.view.add("δ"), step=0, modelName=model.getCurrent(), dataType="NodeData", tags=nodeTags, data=U)
	
	elementTags = arange(1, Nelements+1)
	gmsh.view.addModelData(tag=gmsh.view.add("σx"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=σx)
	gmsh.view.addModelData(tag=gmsh.view.add("σy"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=σy)
	gmsh.view.addModelData(tag=gmsh.view.add("σz"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=σz)
	gmsh.view.addModelData(tag=gmsh.view.add("σxy"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=σxy)
	gmsh.view.addModelData(tag=gmsh.view.add("σyz"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=σyz)
	gmsh.view.addModelData(tag=gmsh.view.add("σzx"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=σzx)

	gmsh.view.addModelData(tag=gmsh.view.add("εx"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=εx)
	gmsh.view.addModelData(tag=gmsh.view.add("εy"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=εy)
	gmsh.view.addModelData(tag=gmsh.view.add("εz"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=εz)
	gmsh.view.addModelData(tag=gmsh.view.add("εxy"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=εxy)
	gmsh.view.addModelData(tag=gmsh.view.add("εyz"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=εyz)
	gmsh.view.addModelData(tag=gmsh.view.add("εzx"), step=0, modelName=model.getCurrent(), dataType="ElementNodeData", tags=elementTags, data=εzx)

	gmsh.fltk.run()
	gmsh.finalize()

if __name__ == '__main__':
	file_name = "cantileverBeam.msh"

	f = 0 			# N
	g = 9.8 		# m/s2
	ρ = 7850 		# kg/m3
	E = 200_000e6 	# N/m2
	ν = 0.3

	properties = {"E" : E,
				  "nu": ν,
				  "bx": 0,
				  "by": -ρ*g,
				  "bz": 0}
	main()