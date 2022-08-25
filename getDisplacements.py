from scipy.sparse.linalg import spsolve
from numpy import zeros, arange, array, setdiff1d, ix_

def deformedShape(Restricted_DOF, Loaded_DOF, NodeTags, F, K, f, DOFS=3):
	U = zeros(DOFS*len(NodeTags))
	Restricted_DOF = array(Restricted_DOF).flatten()
	Loaded_DOF = set(Loaded_DOF)

	for ni in Loaded_DOF:
		F[ni] += f

	free_dof = arange(DOFS*len(NodeTags))
	Restricted_DOF = array(Restricted_DOF)
	free_dof = setdiff1d(free_dof, Restricted_DOF)

	Kff = K[ix_(free_dof, free_dof)]
	Kfc = K[ix_(free_dof, Restricted_DOF)]
	kcf = Kfc.T
	Kcc = K[ix_(Restricted_DOF, Restricted_DOF)]

	ff = F[free_dof]
	fc = F[Restricted_DOF]

	uf = U[free_dof]
	uc = U[Restricted_DOF]

	UF = spsolve(Kff, ff - Kfc @ uc)
	U[free_dof] = UF
	return (U)
