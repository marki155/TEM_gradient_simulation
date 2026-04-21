import numpy as np
from ase.build import graphene
from ase import Atoms
from gpaw import GPAW, PW
import abtem

# ── 1) Atommodell: Graphen-Superzelle mit N-Defekt ────────────────────
def build_graphene_with_N_defect(nx, ny, defect_type='graphitic', seed=42):
    """
    nx, ny       : Superzellen-Wiederholungen
    defect_type  : 'graphitic' | 'pyridinic' | 'pyrrolic'
    """
    atoms = graphene(vacuum=8.0)          # 8 Å Vakuum in z für 2D
    atoms = atoms.repeat([nx, ny, 1])
    atoms.center(vacuum=8.0, axis=2)

    rng = np.random.default_rng(seed)

    if defect_type == 'graphitic':
        # Ein C → N ersetzen
        idx = rng.integers(len(atoms))
        atoms[idx].symbol = 'N'

    elif defect_type == 'pyridinic':
        # Ein C entfernen, Nachbarn → N
        idx = rng.integers(len(atoms))
        del atoms[idx]
        # Nächste Nachbarn finden und einen → N
        dists = atoms.get_distances(0, range(len(atoms)), mic=True)
        neighbor = np.argsort(dists)[1]
        atoms[neighbor].symbol = 'N'

    # Gitterkonstante / Strain anpassen
    # atoms.set_cell(atoms.cell * strain_factor, scale_atoms=True)

    return atoms


# ── 2) DFT-Rechnung mit GPAW ──────────────────────────────────────────
def run_dft(atoms, kpts=(5, 5, 1), ecut=400):
    """
    Führt GPAW-DFT durch und gibt den Calculator zurück.
    Das Potential ist danach direkt abrufbar.
    """
    calc = GPAW(
        mode=PW(ecut),          # Plane-Wave Basis
        kpts=kpts,
        xc='PBE',
        txt='gpaw_output.txt',  # None für kein Log
        convergence={'energy': 1e-6},
    )
    atoms.calc = calc
    atoms.get_potential_energy()   # startet SCF-Zyklus
    return calc


# ── 3) TEM-Simulation mit abTEM ───────────────────────────────────────
def simulate_tem(atoms, calc, defect_type):
    """
    Erzeugt das DFT-Potential über abTEM und simuliert HRTEM/HAADF.
    Die Gittergeometrie kommt aus dem ASE-Atoms-Objekt → vollständig flexibel.
    """
    # DFT-Potential direkt aus GPAW → kein Patch, keine feste Position
    potential = abtem.GPAWPotential(
        calc,
        sampling=0.05,       # Å/px
        slice_thickness=2.0, # Å (für Multislice)
    )

    # Probe definieren
    probe = abtem.Probe(
        energy=80e3,         # 80 kV (typisch für Graphen)
        semiangle_cutoff=20, # mrad
        Cs=0,                # aberrationskorrigiert
    )
    probe.grid.match(potential)

    # HRTEM (kohärent)
    exit_wave = probe.multislice(potential)
    hrtem = exit_wave.intensity()

    # HAADF (alternativ)
    detector = abtem.AnnularDetector(inner=60, outer=200)  # mrad
    haadf = probe.scan(
        scan=abtem.GridScan(potential=potential),
        detectors=detector,
        potential=potential,
    )

    return hrtem, haadf


# ── 4) Hauptprogramm ──────────────────────────────────────────────────
if __name__ == "__main__":

    DEFECT_TYPES = ['graphitic', 'pyridinic']
    STRAIN_VALUES = [0.0, 0.01, 0.02]   # 0%, 1%, 2%

    for defect_type in DEFECT_TYPES:
        for strain in STRAIN_VALUES:

            atoms = build_graphene_with_N_defect(
                nx=5, ny=5,
                defect_type=defect_type,
                seed=42,
            )

            # Strain anwenden – Atompositionen skalieren mit
            if strain != 0.0:
                cell = atoms.get_cell()
                cell[0] *= (1 + strain)
                cell[1] *= (1 + strain)
                atoms.set_cell(cell, scale_atoms=True)

            # DFT
            calc = run_dft(atoms, kpts=(3, 3, 1))

            # TEM
            hrtem, haadf = simulate_tem(atoms, calc, defect_type)

            # Speichern
            tag = f"{defect_type}_strain{strain:.2f}"
            np.save(f"hrtem_{tag}.npy", hrtem.array)
            np.save(f"haadf_{tag}.npy", haadf.array)
            print(f"Fertig: {tag}")