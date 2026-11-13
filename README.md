# AlloyFold — crystal structure prediction

AlloyFold predicts how a molecule packs into a crystal. It generates thousands of symmetric candidate crystals,
minimises each with a physical force field, removes duplicates and ranks the unique structures by lattice energy.
The best candidates can be re-ranked with a machine-learned potential (MACE) and with periodic DFT. Everything runs
on an ordinary multi-core PC, without a GPU: a full search takes about 5–13 minutes per molecule on a laptop CPU.

    python csp.py aspirin --n 1500 --cif out/
    python csp.py caffeine --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --n 1500 --cif out/

## Method

| Step | What happens | Code |
|---|---|---|
| Generate | Random crystals: space group weighted by CSD frequency, cell sized to a sensible packing coefficient, molecules at general positions or on inversion centres; random conformers for flexible molecules. For flat molecules 30 % of the starts are stacked columns (random axis, tilt, spacing; one third alternating) | `generate.py`, `spacegroups.py` |
| Screen | Every candidate minimised at a cheap level (9 Å cutoff, damped shifted-force electrostatics, loose convergence) with a compiled L-BFGS and analytic rigid-body gradients | `lattice.py`, `kernels.py`, `fastmin.py` |
| Adaptive sampling (optional) | Candidates in rounds of 500; stop when a round brings no new structure within 5 kJ/mol of the best (at most 5000) | `csp.py` |
| Deduplicate | Average Minimum Distance (AMD) fingerprints | `invariants.py` |
| Refine | Unique structures within a window of the minimum, at full accuracy (Ewald electrostatics) | `lattice.py` |
| Polish | Lowest structures with flexible molecules (MMFF94 intramolecular strain included) | `flex.py` |
| ML re-rank (optional) | The best 250 re-ranked with the MACE-MPA-0 machine-learned potential (~0.5 s per structure) | `mlrank.py` |
| DFT re-rank (optional) | Periodic PBE-D3(BJ) single points and volume scans for the finalists (PySCF) | `dft_rank.py` |

**Energy models** (`forcefield.py`): MMFF94 via RDKit for organic molecules, with a crystal correction of the
van der Waals term (4 scale factors, `calibrate.py`); TIP4P/Ice for water; TraPPE for CO₂; Sutton-Chen,
Finnis-Sinclair, EAM and MEAM potentials for metals.

Experimental options, off by default (they gave no end-to-end gain so far): contact-based "puzzle" assembly
(`puzzle.py`), coarse sphere/bead pre-packing (`coarse.py`), ellipsoid shaking (`ellipsoid.py`).

## Results

**Blind test.** 9 targets from the CCDC blind tests; the real structures come from the Crystallography Open Database
and were never used for fitting. A target counts as found when a predicted structure matches the force-field-minimised
real crystal (AMD fingerprint distance < 0.005). Adaptive sampling, at most 5000 candidates:

| Target | Found | Force-field rank | ML rank (MACE-MPA-0) |
|---|---|---|---|
| Oxauracil | yes | 1 | 7 |
| Hydantoin | yes | 6 | 3 |
| BT6 XXII | yes | 9 | 2 |
| Hydroxyuracil | yes | 92 | 1 |
| HOAt | yes | 3 | 15 |
| Acrolein (BT4) | yes | 29 | 23 |
| BT4 thione | yes | 171 | 32 |
| DBCFB (BT4) | yes | 194 | 83 |
| BT3 X | no | – | – |

The search finds the real crystal in 8 of 9 cases; in 4 of 9 it is in the ML top 10, the set that would go on to
DFT. All runs: `benchmark_results.jsonl`.

**Compute cost.** About 1–2 CPU-hours per molecule (5–13 minutes on an 8-core laptop), no GPU. For comparison,
published ML-potential workflows report on the order of 200 GPU-hours per molecule, and DFT-based CSP workflows
far more.

**Lattice energies (X23).** Mean absolute error vs diffusion Monte Carlo for the 19 crystals not used in the
calibration: 12.8 kJ/mol (uncalibrated MMFF94: 18.5 kJ/mol). Details: `benchmark_x23.json`, `x23.py`.

**Calibration.** Training: benzene, naphthalene, anthracene, acetic acid, aspirin (experimental densities and
lattice energies). Validation: paracetamol forms I–III. Result: `mmff_crystal_calibration.json`.

## Tools

| Command | Purpose |
|---|---|
| `python csp.py <material> [--smiles S] --n N --cif DIR [--ml mpa0]` | One prediction; top structures as CIF |
| `python csp_benchmark.py [materials] [--adaptive] [--ml mpa0]` | Benchmark against experimental structures |
| `python dft_rank.py <material> [--ml mpa0]` | DFT re-ranking of a benchmark landscape (with `--ml`: of the ML top 10) |
| `python x23.py` | X23 lattice-energy benchmark (structures are downloaded on first use) |
| `python calibrate.py` | Refit the crystal correction |
| `python hunt.py [materials]` | Open-ended polymorph search; `verify_finds.py`, `xrd_finds.py` check finds against the COD |
| `python rerank.py` | Re-minimise saved landscapes with the current force field |

`python csp.py --help` lists all options. Install with `pip install -r requirements.txt` (Python 3.11+).
More cores make it faster: candidates are minimised in parallel.

## License

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): free for research, teaching, personal
use and non-profit organisations. **Commercial use needs a separate license**; please contact
marmorera@gmail.com.

Data and third-party components: [THIRD_PARTY.md](THIRD_PARTY.md).

Required Notice: Copyright (c) 2026 Christoph Deuel, Wittenbach Switzerland
