# AlloyFold — crystal structure prediction

AlloyFold predicts how a molecule packs into a crystal. It generates thousands of symmetric candidate crystals,
minimises each with a physical force field, removes duplicates and ranks the unique structures by lattice energy.
The best candidates can be re-ranked with periodic DFT. Everything runs on an ordinary multi-core PC.

    python csp.py aspirin --n 1500 --cif out/
    python csp.py caffeine --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --n 1500 --cif out/

## Method

| Step | What happens | Code |
|---|---|---|
| Generate | Random crystals: space group weighted by CSD frequency, cell sized to a sensible packing coefficient, molecules at general positions or on inversion centres; random conformers for flexible molecules | `generate.py`, `spacegroups.py` |
| Screen | Every candidate minimised at a cheap level (9 Å cutoff, damped shifted-force electrostatics, loose convergence) | `lattice.py`, `kernels.py` |
| Deduplicate | Average Minimum Distance (AMD) fingerprints | `invariants.py` |
| Refine | Unique structures within a window of the minimum, at full accuracy (Ewald electrostatics) | `lattice.py` |
| Polish | Lowest structures with flexible molecules (MMFF94 intramolecular strain included) | `flex.py` |
| Re-rank (optional) | Periodic PBE-D3(BJ) single points and volume scans for the finalists (PySCF) | `dft_rank.py` |

**Energy models** (`forcefield.py`): MMFF94 via RDKit for organic molecules, with a crystal correction of the
van der Waals term (4 scale factors, `calibrate.py`); TIP4P/Ice for water; TraPPE for CO₂; Sutton-Chen,
Finnis-Sinclair, EAM and MEAM potentials for metals.

Experimental options, off by default (they gave no end-to-end gain so far): contact-based "puzzle" assembly
(`puzzle.py`), coarse sphere/bead pre-packing (`coarse.py`), ellipsoid shaking (`ellipsoid.py`).

## Results

**Blind test.** 9 targets from the CCDC blind tests; the real structures come from the Crystallography Open Database
and were never used for fitting. Rank of the real crystal among the predictions (force field only, 1500 candidates; acrolein 4000):

| Target | Rank | ΔE to #1 (kJ/mol) |
|---|---|---|
| Oxauracil | 1 | 0 |
| Hydantoin | 4 | 3.2 |
| HOAt | 15–22 | 5.8 |
| BT6 XXII | 19–21 | 9.4 |
| Acrolein (BT4) | 22 | 1.9 |
| BT4 thione | 32–36 | 8.5–11.2 |
| Hydroxyuracil | 38 | 16.1 |
| BT3 X | not found | |
| DBCFB (BT4) | not found | |

The search finds the real crystal in 7 of 9 cases; the ranking is the weak point (force-field errors of several
kJ/mol). DFT re-ranking so far: oxauracil, real crystal #3 of 40 after PBE-D3. All runs: `benchmark_results.jsonl`.

**Lattice energies (X23).** Mean absolute error vs diffusion Monte Carlo for the 19 crystals not used in the
calibration: 12.8 kJ/mol (uncalibrated MMFF94: 18.5 kJ/mol). Details: `benchmark_x23.json`, `x23.py`.

**Calibration.** Training: benzene, naphthalene, anthracene, acetic acid, aspirin (experimental densities and
lattice energies). Validation: paracetamol forms I–III. Result: `mmff_crystal_calibration.json`.

## Tools

| Command | Purpose |
|---|---|
| `python csp.py <material> [--smiles S] --n N --cif DIR` | One prediction; top structures as CIF |
| `python csp_benchmark.py [materials]` | Benchmark against experimental structures |
| `python dft_rank.py <material>` | DFT re-ranking of a benchmark landscape |
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
