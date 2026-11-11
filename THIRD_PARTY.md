# Data and third-party components

AlloyFold's own code is under the PolyForm Noncommercial License (LICENSE.md). The following parts come from
others and keep their own terms. Please cite them when you publish results that use them.

## Data in this repository

| Files | Source | Terms |
|---|---|---|
| `references/cod_*.cif`, `references/cod_index_*.json` | [Crystallography Open Database](https://www.crystallography.net/cod/) (each CIF names its authors and publication) | Public domain (CC0) |
| `references/potentials/` | [NIST Interatomic Potentials Repository](https://www.ctcms.nist.gov/potentials/): Zope & Mishin, Phys. Rev. B 68, 024102 (2003); Fereidonnejad et al., Comput. Mater. Sci. 213, 111685 (2022); see `SOURCE.txt` | Cite the original papers |

## Data downloaded when needed (not included)

| What | Source |
|---|---|
| X23 benchmark structures (`x23.py`) | [MolCrys-MACE](https://github.com/water-ice-group/MolCrys-MACE), Della Pia et al., Chem. Sci. 2025 |
| Further experimental structures (`references.py`) | Crystallography Open Database |

## Methods and reference values used in the code

MMFF94 (Halgren 1996, via RDKit); TIP4P/Ice (Abascal et al. 2005); TraPPE CO₂ (Potoff & Siepmann 2001);
Sutton-Chen (1990) and Finnis-Sinclair (1984) metal potentials; damped shifted-force electrostatics (Fennell &
Gezelter 2006); AMD fingerprints (Widdowson et al. 2022); X23 lattice energies from Della Pia et al., Phys. Rev.
Lett. 2024, and Dolgonos, Hoja & Boese 2019 (X23b); PBE-D3(BJ) via PySCF and simple-dftd3.

## Python packages (installed separately, not included)

NumPy, SciPy, PyTorch, Numba, spglib, psutil (BSD-style licenses); RDKit (BSD); ASE (LGPL); optional PySCF
(Apache 2.0) and simple-dftd3 (LGPL). None of these restricts how AlloyFold itself is licensed.
