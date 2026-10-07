"""Headless-Test fuer das zufaellige Atom-Seeding (``atom_seeding.py``).

Das Zielbild wird mit demselben Modelltyp aus bekannten Graphen-Positionen
gerendert; der Seeder startet ohne Atome und soll die Positionen wieder
finden.

Aufruf:  python3 test_atom_seeding.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multislice

multislice.CHECK_PHASE = False

from atom_seeding import AtomStore, RandomAtomSeeder
from mains.main_random_atoms import build_seeding_model
from nn import CONFIG_CLASS_REGISTRY
from mains.main import deserialize

SHAPE = (96, 96)
PIXEL_NM = 0.02          # 0.2 A/px: Bindung ~7.1 px, touch 2.5 px, merge 5 px
TEM_ABERRATIONS = {"C10": -5.0, "C30": 4.16}


def graphene_lattice(shape, pixel_nm):
    a = 2.46 / (pixel_nm * 10)
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2])
    basis = np.array([a / 2, a / (2 * np.sqrt(3))])
    points = []
    for i in range(-30, 60):
        for j in range(-30, 60):
            p = i * a1 + j * a2
            points += [p, p + basis]
    points = np.array(points) + np.array([3.3, 2.1])
    keep = ((points[:, 0] > 2) & (points[:, 0] < shape[0] - 3)
            & (points[:, 1] > 2) & (points[:, 1] < shape[1] - 3))
    return points[keep]


def make(mode, aberrations=None, border_px=0):
    image = np.random.default_rng(0).random(SHAPE)
    return build_seeding_model(image, PIXEL_NM, mode, border_px=border_px,
                               aberrations=aberrations, padding=20)


def render_target(mode, true, aberrations=None):
    model, _t, _tm = make(mode, aberrations)
    AtomStore(model.potential_layer).apply([], true)
    with torch.no_grad():
        image = model.model(None)[0].detach().clone()
    return (image - image.min()) / (image.max() - image.min())


def recovery(found, true, tolerance_px=2.5):
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(found).query(true)
    return float((distance < tolerance_px).mean())


def test_store_starts_empty_and_grows():
    model, target, _ = make("stem")
    pot = model.potential_layer
    objects = [id(p) for p in pot.parameters()]

    store = AtomStore(pot)
    assert store.n == 0
    with torch.no_grad():
        image = model.model(None)[0]
    assert torch.isfinite(image).all() and float(image.abs().max()) == 0.0

    token = store.apply([], [[10, 10], [30, 30], [50, 50]])
    assert store.n == 3 and pot.get_init_positions(0).shape[0] == 3
    with torch.no_grad():
        assert float(model.model(None)[0].max()) > 0
    store.apply([1], [[70, 70]])
    assert store.n == 3 and np.allclose(store.positions, [[10, 10], [50, 50], [70, 70]])
    assert [id(p) for p in pot.parameters()] == objects, "Parameter-Objekte muessen gleich bleiben"

    other = AtomStore(pot)
    assert np.allclose(other.positions, store.positions), "Zustand wird aus dem Modell gelesen"
    store.undo(token)
    print("ok  store: leer startet, waechst/schrumpft, Parameter-Objekte bleiben")


def test_store_sync_drops_unoccupied():
    model, _, _ = make("stem")
    store = AtomStore(model.potential_layer)
    store.apply([], [[10, 10], [30, 30], [50, 50]])
    with torch.no_grad():
        store.tensors()[4][1] = -5.0
    store.sync()
    assert store.n == 2 and np.allclose(store.positions, [[10, 10], [50, 50]])
    print("ok  store: Atome mit Occupancy < 0.5 fallen beim Einlesen weg")


def seeder_with(atoms, mode="stem", **kwargs):
    model, target, _ = make(mode)
    seeder = RandomAtomSeeder(model, target, **kwargs)
    seeder.atoms.apply([], atoms)
    seeder.resync(sync_store=False)
    return seeder


def test_collision_rule_matches_script():
    # 0.2 A/px: touch 0.5 A = 2.5 px, merge 1 A = 5 px
    seeder = seeder_with(np.array([[50.0, 50.0], [50.0, 80.0]]))

    (x, y), grave, merged = seeder.resolve_collisions(20.0, 20.0)
    assert (x, y) == (20.0, 20.0) and grave == [] and not merged, "weit weg: unveraendert"

    (x, y), grave, merged = seeder.resolve_collisions(51.0, 51.0)
    assert (x, y) == (51.0, 51.0) and grave == [0] and not merged, "beruehrt: Nachbar stirbt, Position bleibt"

    (x, y), grave, merged = seeder.resolve_collisions(50.0, 54.0)
    assert np.allclose([x, y], [50.0, 52.0]) and grave == [0] and merged, \
        "nah: Mittelpunkt, Nachbar stirbt"

    (x, y), grave, merged = seeder.resolve_collisions(50.0, 75.0)
    assert np.allclose([x, y], [50.0, 77.5]) and grave == [1] and merged
    print("ok  Kollisionsregel wie im Skript")


def test_collision_chain_terminates():
    chain = np.array([[50.0, 50.0 + 4.0 * i] for i in range(8)])
    seeder = seeder_with(chain)
    (x, y), grave, _ = seeder.resolve_collisions(50.0, 52.0)
    assert len(set(grave)) == len(grave) and len(grave) >= 1
    others = [c for i, c in enumerate(chain) if i not in grave]
    for c in others:
        assert np.hypot(c[0] - x, c[1] - y) > seeder.touch_px
    print("ok  Kollisionskette terminiert")


def test_rejected_trial_restores_everything():
    true = graphene_lattice(SHAPE, PIXEL_NM)[:30]
    for local in (False, True):
        seeder = seeder_with(true, move_prob=0.3, remove_prob=0.2, seed=3, local=local)
        seeder.loss = -np.inf          # nichts kann mehr besser sein
        before = seeder.atoms.positions.copy()
        with torch.no_grad():
            image_before = seeder.model.model(None)[0].clone()

        for _ in range(40):
            assert seeder.trial() is False

        assert np.array_equal(seeder.atoms.positions, before)
        with torch.no_grad():
            assert torch.equal(seeder.model.model(None)[0], image_before), \
                "Bild muss bitgleich zurueck sein"
        mirror = seeder.atoms.positions.copy()
        seeder.atoms.sync()
        assert np.allclose(seeder.atoms.positions, mirror)
    print("ok  abgelehnte Versuche (Einfuegen/Verschieben/Entfernen) stellen alles exakt her")


def test_neighbor_distribution():
    # 0.2 A/px: Bindung 1.42 A = 7.1 px, sigma 0.2 A = 1 px, Cutoffs 2.5 .. 10 px
    seeder = seeder_with(np.array([[48.0, 48.0]]))
    seeder.neighbor_seeding = True
    points = np.array([seeder._neighbor_point() for _ in range(4000)])
    radius = np.hypot(points[:, 0] - 48.0, points[:, 1] - 48.0)
    assert radius.min() >= 2.5 - 1e-9 and radius.max() <= 10.0 + 1e-9, "Cutoffs"
    assert abs(radius.mean() - 7.1) < 0.1 and abs(radius.std() - 1.0) < 0.1, \
        f"Gauss um die Bindungslaenge: {radius.mean():.2f} +- {radius.std():.2f} px"
    angle = np.arctan2(points[:, 1] - 48.0, points[:, 0] - 48.0)
    assert abs(np.cos(angle).mean()) < 0.05 and abs(np.sin(angle).mean()) < 0.05, "Winkel gleichverteilt"
    assert abs(seeder.bond_A - 1.42) < 1e-9 and abs(seeder.neighbor_max_A - 2.0) < 1e-9

    seeder.margin_px = 45.0          # Platz nur noch 45..51: Punkte ausserhalb werden verworfen
    inside = [seeder._neighbor_point() for _ in range(200)]
    assert all(p is None or (45.0 <= p[0] < 51.0 and 45.0 <= p[1] < 51.0) for p in inside)

    seeder.margin_px = 0.0
    atoms = np.array([[20.0, 20.0], [20.0, 70.0], [70.0, 70.0], [70.0, 20.0], [45.0, 45.0]])
    ring = seeder_with(atoms)
    ring.neighbor_seeding = True
    order = []
    for _ in range(12):
        x, y = ring._neighbor_point()
        order.append(int(np.argmin(np.hypot(atoms[:, 0] - x, atoms[:, 1] - y))))
    assert order == [0, 1, 2, 3, 4] * 2 + [0, 1], f"Reihenfolge {order}"
    skipped = [ring._neighbor_point(exclude=[ring._neighbor_cursor % 5]) for _ in range(3)]
    assert all(p is not None for p in skipped)

    empty = seeder_with(np.zeros((0, 2)))
    empty.neighbor_seeding = True
    assert empty._neighbor_point() is None
    xy, grave, merged = empty.propose()
    assert grave == [] and not merged, "ohne Atome: gleichverteilt"
    print(f"ok  Nachbar-Positionen: r = {radius.mean():.2f} +- {radius.std():.2f} px "
          f"[{radius.min():.1f}, {radius.max():.1f}]")


def test_neighbor_seeding_grows_connected():
    model, target, _ = make("stem")
    seeder = RandomAtomSeeder(model, target, seed=3, margin_px=4, neighbor_seeding=True, max_atoms=12)
    seeder.run(600, log_every=0)
    pos = seeder.atoms.positions
    assert len(pos) >= 3
    first = seeder.pixel_A
    dist = np.hypot(*(pos[:, None] - pos[None]).transpose(2, 0, 1)) + np.eye(len(pos)) * 1e9
    assert (dist.min(axis=1) * first <= 2.5).all(), "jedes Atom hat einen Nachbarn in Bindungsnaehe"
    print(f"ok  Nachbar-Seeding: {len(pos)} Atome, groesster Nachbarabstand {dist.min(axis=1).max() * first:.2f} A")


def test_max_atoms_stops_cleanly():
    model, target, _ = make("stem")
    seeder = RandomAtomSeeder(model, target, seed=0, max_atoms=6)
    stats = seeder.run(400, log_every=0)
    assert seeder.n_atoms <= 6 and (stats.full or seeder.n_atoms < 6)
    print(f"ok  Atomgrenze: {seeder.n_atoms}/6 Atome, full={stats.full}")


def check_local_matches_full(mode, aberrations, trials, tolerance):
    """Der lokale Simulator muss Bild und Loss der vollen Simulation treffen."""

    true = graphene_lattice(SHAPE, PIXEL_NM)
    target = render_target(mode, true, aberrations)
    model, _t, _tm = make(mode, aberrations, border_px=4)
    seeder = RandomAtomSeeder(model, target, seed=1, margin_px=4,
                              move_prob=0.1, remove_prob=0.05, refresh_every=0)
    assert seeder.uses_local, seeder.local_note
    worst_loss = worst_image = 0.0
    for k in range(trials):
        seeder.trial()
        if k % 25 == 24:
            full = seeder.evaluate()
            with torch.no_grad():
                image = model.model(None)[0]
            scale = float(image.abs().max())
            worst_loss = max(worst_loss, abs(seeder.loss - full))
            worst_image = max(worst_image, float((seeder.local.sim - image).abs().max()) / scale)
    assert seeder.stats.accepted > 20 and seeder.stats.fallbacks == 0
    print(f"ok  lokal == voll ({mode}): {seeder.stats.accepted} angenommen, "
          f"max |dLoss| {worst_loss:.1e}, max |dBild| {worst_image:.1e}")
    assert worst_loss < tolerance and worst_image < tolerance


def test_local_matches_full_stem():
    check_local_matches_full("stem", None, 400, 2e-3)


def test_local_matches_full_tem():
    check_local_matches_full("tem", TEM_ABERRATIONS, 400, 2e-3)


def test_local_and_full_find_same_decisions():
    true = graphene_lattice(SHAPE, PIXEL_NM)
    target = render_target("stem", true)
    runs = []
    for local in (False, True):
        model, _t, _tm = make("stem", border_px=4)
        seeder = RandomAtomSeeder(model, target, seed=5, margin_px=4, local=local)
        seeder.run(300, log_every=0)
        runs.append((seeder.n_atoms, seeder.loss))
    print(f"ok  gleiche Entscheidungen: voll {runs[0][0]} Atome / lokal {runs[1][0]} Atome")
    assert abs(runs[0][0] - runs[1][0]) <= max(3, 0.1 * runs[0][0])
    assert abs(runs[0][1] - runs[1][1]) < 0.02


def test_stem_recovers_lattice():
    true = graphene_lattice(SHAPE, PIXEL_NM)
    target = render_target("stem", true)

    model, _t, _tm = make("stem")
    seeder = RandomAtomSeeder(model, target, seed=1)
    first = None
    for _ in range(5):
        seeder.run(500, log_every=0)
        first = first if first is not None else seeder.loss
    found = seeder.positions_px()
    rate = recovery(found, true)
    print(f"ok  STEM: {len(found)}/{len(true)} Atome, {rate:.0%} innerhalb 2.5 px, "
          f"Loss {first:.3f} -> {seeder.loss:.3f}")
    assert rate >= 0.8, rate
    assert 0.8 * len(true) <= len(found) <= 1.2 * len(true)
    assert seeder.loss < first
    return seeder


def test_tem_makes_progress():
    true = graphene_lattice(SHAPE, PIXEL_NM)
    target = render_target("tem", true, TEM_ABERRATIONS)

    model, _t, _tm = make("tem", TEM_ABERRATIONS)
    seeder = RandomAtomSeeder(model, target, seed=1)
    seeder.run(10, log_every=0)
    early = seeder.loss
    seeder.run(2500, log_every=0)
    found = seeder.positions_px()
    rate = recovery(found, true)
    print(f"ok  TEM: {len(found)}/{len(true)} Atome, {rate:.0%} innerhalb 2.5 px, "
          f"Loss {early:.3f} -> {seeder.loss:.3f}")
    assert seeder.loss < early
    assert rate >= 0.35, rate


def test_tune_improves_wrong_probe():
    true = graphene_lattice(SHAPE, PIXEL_NM)
    target = render_target("stem", true)

    model, _t, train_method = make("stem")
    train_method.train_ctf_aberrations.lr = 0.05
    seeder = RandomAtomSeeder(model, target, seed=0)
    seeder.atoms.apply([], true)
    seeder.resync(sync_store=False)
    good = seeder.loss

    defocus = model.imaging_layer.tunable_parameters.aberrations["C10"]
    with torch.no_grad():
        defocus += 4.0
    seeder.resync(sync_store=False)
    bad = seeder.loss
    assert bad > good + 1e-3, (bad, good)

    seeder.tune(60)
    after = seeder.loss
    print(f"ok  tune: Sonde falsch {bad:.4f} -> {after:.4f} (Soll {good:.4f}), C10 {float(defocus):.2f}")
    assert after < bad
    assert np.allclose(seeder.atoms.positions, true), "tune darf Atome nicht anfassen"


def test_gradient_landscape_matches_local_loss():
    true = graphene_lattice(SHAPE, PIXEL_NM)
    target = render_target("stem", true)
    model, _t, _tm = make("stem")
    seeder = RandomAtomSeeder(model, target, seed=1, margin_px=4)
    for _ in range(60):
        seeder.trial()
    local = seeder.local
    gain = local.insertion_gain(seeder.atoms.new_weight)
    rng = np.random.default_rng(0)
    predicted, actual = [], []
    for _ in range(80):
        i, j = rng.integers(6, 90, 2)
        predicted.append(gain[i, j])
        actual.append(local.propose([], np.array([[float(i), float(j)]]), seeder.atoms.new_weight).loss - seeder.loss)
    predicted, actual = np.array(predicted), np.array(actual)
    assert np.abs(predicted - actual).max() < 1e-5, np.abs(predicted - actual).max()

    seeder.gradient_seeding = True
    seeder._version += 1
    masked = seeder._gain_map()
    finite = np.isfinite(masked)
    assert finite.any() and (masked[finite] < 0).all(), "nur Verbesserungen bleiben"
    rows, cols = np.nonzero(finite)
    distance = np.hypot(rows[:, None] - seeder.atoms.positions[None, :, 0],
                        cols[:, None] - seeder.atoms.positions[None, :, 1]).min(axis=1)
    assert (distance > seeder.merge_px - 0.75).all(), "keine Stellen in der Kollisionszone"
    print(f"ok  Gradienten-Landschaft exakt (max Abweichung {np.abs(predicted - actual).max():.1e}), "
          f"{finite.sum()} Stellen verbessern, keine in der Kollisionszone")


def test_gradient_seeding_beats_random():
    true = graphene_lattice(SHAPE, PIXEL_NM)
    target = render_target("stem", true)
    losses = {}
    for name, kw in (("zufall", {}), ("gradient", dict(gradient_seeding=True)),
                     ("gradient+nachbar", dict(gradient_seeding=True, neighbor_seeding=True))):
        model, _t, _tm = make("stem")
        seeder = RandomAtomSeeder(model, target, seed=1, margin_px=4, **kw)
        for _ in range(200):
            seeder.trial()
        losses[name] = seeder.loss
        if name != "zufall":
            assert (seeder.atoms.positions >= 4 - 1e-6).all() and (seeder.atoms.positions <= 96 - 4).all()
    assert losses["gradient"] < losses["zufall"] - 0.1 and losses["gradient+nachbar"] < losses["zufall"] - 0.05, losses
    print("ok  nach 200 Versuchen: " + ", ".join(f"{k} {v:.3f}" for k, v in losses.items()))


def test_gradient_ignored_for_tem():
    model, target, _ = make("tem")
    seeder = RandomAtomSeeder(model, target, seed=0, margin_px=4, gradient_seeding=True)
    assert not seeder.gradient_supported
    seeder.run(30, log_every=0)
    print("ok  Gradienten-Seeding im TEM: wird uebergangen, normales Seeding")


def test_save_roundtrip():
    true = graphene_lattice(SHAPE, PIXEL_NM)[:60]
    model, target, _ = make("stem")
    seeder = RandomAtomSeeder(model, target, seed=0)
    seeder.atoms.apply([], true)
    with torch.no_grad():
        reference = model.model(None)[0].detach().clone()

    with tempfile.TemporaryDirectory() as folder:
        seeder.save(folder, "t")
        assert os.path.exists(os.path.join(folder, "t.xyz"))
        with open(os.path.join(folder, "t_config.json")) as f:
            config = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)

    assert len(config.init_positions[0]) == len(true)

    from nn import NN_particles
    from training_parameter import TrainMethod
    tm = TrainMethod().ensure_for_config(config)
    reloaded = NN_particles(config=config, pixel_size_nm=config.pixel_size_nm,
                            voltage_kV=config.voltage_kV, train_method=tm,
                            device=model.device, train_methods=tm)
    reloaded.build()
    with torch.no_grad():
        again = reloaded.model(None)[0]
    scale = float(reference.abs().max())
    assert float((again - reference).abs().max()) < 1e-4 * scale
    print("ok  gespeicherte config.json ergibt dasselbe Bild")


def test_dataset_folder():
    """Ordner mit target.tif + mask_Gr.tif -> config_random/ mit config, train_method, ..."""

    import tifffile
    from mains import main_random_atoms as mra
    from training_parameter import load_train_method

    true = graphene_lattice(SHAPE, PIXEL_NM)
    true = true[true[:, 1] < 60]                       # Atome nur links
    image = render_target("stem", true).cpu().numpy()
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[:, :64] = 255

    with tempfile.TemporaryDirectory() as folder:
        tifffile.imwrite(os.path.join(folder, "target.tif"), image.astype(np.float32))
        tifffile.imwrite(os.path.join(folder, "mask_Gr.tif"), mask)

        args = [folder, "--pixel-size-nm", str(PIXEL_NM), "--trials", "1500", "--border-px", "4",
                "--tune-every", "0", "--log-every", "0", "--seed", "1"]
        assert mra.main(args) == 0

        out = os.path.join(folder, "config_random")
        for name in ("config.json", "train_method.json", "target.npy", "seeded.xyz",
                     "seeded_positions_px.npy", "seeded_preview.png", "area_masks.npz"):
            assert os.path.exists(os.path.join(out, name)), name
        assert os.path.exists(os.path.join(folder, "init_random.json"))

        with open(os.path.join(out, "config.json")) as f:
            config = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)
        assert config.group_names == ["Gr"] and config.groups[0].mask_file == "mask_Gr.tif"
        positions = config.init_positions[0]
        assert len(positions) > 0.5 * len(true)
        assert (positions[:, 1] < 65).all(), "Atome nur innerhalb der Maske"

        tm = load_train_method(os.path.join(out, "train_method.json"), config)
        assert int(tm.masks["Gr"].sum()) == 64 * SHAPE[0], (int(tm.masks["Gr"].sum()), tm.masks["Gr"].dtype)
        assert tm.layer("Gr").free_positions.enabled

        # zweiter Lauf: nur der Ordner, Atome aus dem ersten Lauf bleiben
        before = len(positions)
        assert mra.main([folder, "--trials", "200", "--border-px", "4",
                         "--tune-every", "0", "--log-every", "0", "--seed", "2"]) == 0
        with open(os.path.join(out, "config.json")) as f:
            again = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)
        assert len(again.init_positions[0]) >= before * 0.9, "vorhandener Stand wird automatisch weitergefuehrt"

        # --fresh: neu anfangen, alter Stand in Sicherungsordner
        assert mra.main([folder, "--fresh", "--trials", "5", "--border-px", "4",
                         "--tune-every", "0", "--log-every", "0", "--seed", "2"]) == 0
        backups = [d for d in os.listdir(folder) if d.startswith("config_random_")]
        assert len(backups) == 1
    print(f"ok  Datenordner: {before} Atome, Resume {len(again.init_positions[0])}")


def main():
    test_store_starts_empty_and_grows()
    test_store_sync_drops_unoccupied()
    test_collision_rule_matches_script()
    test_collision_chain_terminates()
    test_rejected_trial_restores_everything()
    test_max_atoms_stops_cleanly()
    test_neighbor_distribution()
    test_gradient_landscape_matches_local_loss()
    test_gradient_seeding_beats_random()
    test_gradient_ignored_for_tem()
    test_neighbor_seeding_grows_connected()
    test_local_matches_full_stem()
    test_local_matches_full_tem()
    test_local_and_full_find_same_decisions()
    test_save_roundtrip()
    test_dataset_folder()
    test_tune_improves_wrong_probe()
    test_stem_recovers_lattice()
    test_tem_makes_progress()
    print("\nalle Seeding-Tests bestanden")


if __name__ == "__main__":
    main()
