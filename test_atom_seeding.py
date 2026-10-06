"""Headless-Test fuer das zufaellige Atom-Seeding (``atom_seeding.py``).

Das Zielbild wird mit demselben Modelltyp aus bekannten Graphen-Positionen
gerendert; der Seeder startet mit leerem Pool und soll die Positionen wieder
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

from atom_seeding import (
    INACTIVE_OCCUPANCY, PARK_POSITION, AtomPool, RandomAtomSeeder, graphene_capacity)
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


def make(mode, capacity, aberrations=None, border_px=0):
    image = np.random.default_rng(0).random(SHAPE)
    return build_seeding_model(image, PIXEL_NM, mode, capacity, border_px=border_px,
                               aberrations=aberrations, padding=20)


def render_target(mode, true, capacity, aberrations=None):
    model, _t, _tm = make(mode, capacity, aberrations)
    pool = AtomPool(model.potential_layer)
    pool.set_slots(np.arange(len(true)), true, True)
    with torch.no_grad():
        image = model.model(None)[0].detach().clone()
    return (image - image.min()) / (image.max() - image.min())


def recovery(found, true, tolerance_px=2.5):
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(found).query(true)
    return float((distance < tolerance_px).mean())


def test_pool_starts_empty_and_shapes_stay_fixed():
    model, target, _ = make("stem", 50)
    pot = model.potential_layer
    parameters = {name: p.shape for name, p in pot.named_parameters()}
    objects = [id(p) for p in pot.parameters()]

    pool = AtomPool(pot)
    assert pool.n_active == 0 and pool.free_slot() == 0
    with torch.no_grad():
        image = model.model(None)[0]
    assert torch.isfinite(image).all() and float(image.abs().max()) == 0.0, "geparkte Atome duerfen nichts malen"

    pool.set_slots([0, 1, 2], [[10, 10], [30, 30], [50, 50]], True)
    assert pool.n_active == 3 and pool.free_slot() == 3
    with torch.no_grad():
        assert float(model.model(None)[0].max()) > 0

    assert {n: p.shape for n, p in pot.named_parameters()} == parameters
    assert [id(p) for p in pot.parameters()] == objects, "Parameter-Objekte muessen gleich bleiben"
    print("ok  pool: leer startet, Formen und Parameter-Objekte bleiben")


def test_pool_sync_reads_model_state():
    model, _, _ = make("stem", 20)
    pool = AtomPool(model.potential_layer)
    pool.set_slots([2, 5], [[20, 20], [40, 40]], True)

    other = AtomPool(model.potential_layer)
    assert other.n_active == 2
    assert np.allclose(other.active_positions(), [[20, 20], [40, 40]])
    print("ok  pool: Zustand wird aus dem Modell gelesen")


def seeder_with(atoms, **kwargs):
    model, target, _ = make("stem", 40)
    seeder = RandomAtomSeeder(model, target, **kwargs)
    seeder.pool.set_slots(np.arange(len(atoms)), atoms, True)
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
    seeder = seeder_with(true, move_prob=0.3, remove_prob=0.2, seed=3)
    seeder.loss = -np.inf          # nichts kann mehr besser sein
    before = (seeder.pool.positions.copy(), seeder.pool.active.copy())
    with torch.no_grad():
        image_before = seeder.model.model(None)[0].clone()

    for _ in range(40):
        assert seeder.trial() is False

    assert np.array_equal(seeder.pool.active, before[1])
    assert np.array_equal(seeder.pool.positions, before[0])
    with torch.no_grad():
        assert torch.equal(seeder.model.model(None)[0], image_before), "Bild muss bitgleich zurueck sein"
    # Spiegel stimmt mit dem Modell ueberein
    mirror = (seeder.pool.positions.copy(), seeder.pool.active.copy())
    seeder.pool.sync()
    assert np.array_equal(seeder.pool.active, mirror[1])
    assert np.allclose(seeder.pool.positions[seeder.pool.active], mirror[0][mirror[1]])
    print("ok  abgelehnte Versuche (Einfuegen/Verschieben/Entfernen) stellen alles exakt her")


def test_pool_full_stops_cleanly():
    model, target, _ = make("stem", 6)
    seeder = RandomAtomSeeder(model, target, seed=0)
    stats = seeder.run(400, log_every=0)
    assert seeder.pool.n_active <= 6
    assert stats.full or seeder.pool.n_active < 6
    print(f"ok  voller Pool: {seeder.pool.n_active}/6 Atome, full={stats.full}")


def test_stem_recovers_lattice():
    true = graphene_lattice(SHAPE, PIXEL_NM)
    capacity = int(len(true) * 1.5)
    target = render_target("stem", true, capacity)

    model, _t, _tm = make("stem", capacity)
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
    capacity = int(len(true) * 1.5)
    target = render_target("tem", true, capacity, TEM_ABERRATIONS)

    model, _t, _tm = make("tem", capacity, TEM_ABERRATIONS)
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
    capacity = int(len(true) * 1.5)
    target = render_target("stem", true, capacity)

    model, _t, train_method = make("stem", capacity)
    train_method.train_ctf_aberrations.lr = 0.05
    seeder = RandomAtomSeeder(model, target, seed=0)
    seeder.pool.set_slots(np.arange(len(true)), true, True)
    seeder.loss = seeder.evaluate()
    good = seeder.loss

    defocus = model.imaging_layer.tunable_parameters.aberrations["C10"]
    with torch.no_grad():
        defocus += 4.0
    bad = seeder.evaluate()
    assert bad > good + 1e-3, (bad, good)

    seeder.tune(60)
    after = seeder.loss
    print(f"ok  tune: Sonde falsch {bad:.4f} -> {after:.4f} (Soll {good:.4f}), C10 {float(defocus):.2f}")
    assert after < bad
    assert np.array_equal(seeder.pool.active, np.arange(capacity) < len(true)), "tune darf Atome nicht anfassen"


def test_save_roundtrip():
    true = graphene_lattice(SHAPE, PIXEL_NM)[:60]
    model, target, _ = make("stem", 100)
    seeder = RandomAtomSeeder(model, target, seed=0)
    seeder.pool.set_slots(np.arange(len(true)), true, True)
    with torch.no_grad():
        reference = model.model(None)[0].detach().clone()

    with tempfile.TemporaryDirectory() as folder:
        seeder.save(folder, "t")
        assert os.path.exists(os.path.join(folder, "t.xyz"))
        with open(os.path.join(folder, "t_config.json")) as f:
            config = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)

    assert len(config.init_positions[0]) == len(true)
    assert seeder.model.config.init_positions[0].shape[0] == 100, "Modell-Config bleibt unveraendert"

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
    capacity = int(len(true) * 2)
    image = render_target("stem", true, capacity).cpu().numpy()
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
        assert mra.main([folder, "--resume", "--trials", "200", "--border-px", "4",
                         "--tune-every", "0", "--log-every", "0", "--seed", "2"]) == 0
        with open(os.path.join(out, "config.json")) as f:
            again = deserialize(json.load(f), CONFIG_CLASS_REGISTRY)
        assert len(again.init_positions[0]) >= before * 0.9
    print(f"ok  Datenordner: {before} Atome, Resume {len(again.init_positions[0])}")


def test_graphene_capacity():
    assert abs(graphene_capacity((100, 100), 0.1, 1.0) - 3820) <= 1 and abs(graphene_capacity((100, 100), 0.1, 1.5) - 5730) <= 1
    print("ok  Pool-Groesse")


def main():
    test_graphene_capacity()
    test_pool_starts_empty_and_shapes_stay_fixed()
    test_pool_sync_reads_model_state()
    test_collision_rule_matches_script()
    test_collision_chain_terminates()
    test_rejected_trial_restores_everything()
    test_pool_full_stops_cleanly()
    test_save_roundtrip()
    test_dataset_folder()
    test_tune_improves_wrong_probe()
    test_stem_recovers_lattice()
    test_tem_makes_progress()
    print("\nalle Seeding-Tests bestanden")


if __name__ == "__main__":
    main()
