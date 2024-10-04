import pyscf
import scipy.linalg
import numpy as np

# need structure information of molecule and electronic integrals 
mol = pyscf.gto.M(atom='H 0.6 0 0; H -0.6 0 0', basis='sto-3g')
mf = pyscf.scf.RHF(mol)
mf.kernel()
print(f'Reference HF total energy = {mf.e_tot}')

# RHF.
S = mol.intor('int1e_ovlp')  # Overlap integrals
H_core = mol.intor('int1e_kin') + mol.intor('int1e_nuc') # Kinetic energy integrals + Nuclear attraction integrals = Core Hamiltonian

nao = H_core.shape[0] # Number of atomic orbitals
eri = mol.intor('int2e').reshape(nao,nao,nao,nao) # Electron repulsion integrals
nocc = mol.nelectron // 2  # Number of occupied orbitals

# calculate E_nuc
def energy_nuc(mol):

    charges = mol.atom_charges()
    coords = mol.atom_coords() 

    rr = np.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)  # N * N matrix 
    rr[np.diag_indices_from(rr)] = 1e200 # avoid denominator = 0
    e_nuc = np.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e_nuc

# get fock matrix
def get_dm(fock):

    mo_energy, mo_coeff = scipy.linalg.eigh(fock, S) # Generalized eigenvector
    
    #Sort the energy values and obtain the sorted indices:
    e_idx = np.argsort(mo_energy) 
    #Sort energy and coefficients by sorted indices:
    mo_energy = mo_energy[e_idx] 
    mo_coeff = mo_coeff[:, e_idx]
    #Select the coefficient matrix of the occupied orbitals:
    occupied_mo_coeff = mo_coeff[:, :nocc]

    dm = np.dot(occupied_mo_coeff, occupied_mo_coeff.T) 
    return dm

#initialize
dm= np.eye(2,2) #initial guess
#dm = get_dm(H_core)
scf_conv = False
cycle = 0
e_tot = 0

while not scf_conv and cycle < 50:
    dm_last = dm
    last_hf_e = e_tot
    fock = H_core + np.einsum('pqrs,rs->pq', eri, dm) * 2 - np.einsum('psrq,rs->pq', eri, dm)      
    dm = get_dm(fock)

    e_tot = np.einsum('ij,ji->', H_core + fock, dm)  + energy_nuc(mol)

    norm_ddm = np.linalg.norm(dm-dm_last)
    if abs(e_tot-last_hf_e) < 1.0E-8 and norm_ddm < 1.0E-6:

        scf_conv = True
    
    cycle += 1

print(f'RHF total energy = {e_tot}', f'Cycle number={cycle}')

#verify
mf = pyscf.scf.RHF(mol)
mf.scf(dm0=dm)
print(mf.e_tot)
print(np.allclose(mf.e_tot, e_tot))






