"""Part IV, Section 3: jobs and pacing."""

import pytest

from temsim.service.jobs import Clock, SeedingJob, TrainingJob
from temsim.service.session import Session, SessionError

from .helpers import make_folder, make_run


def test_clock_follows_the_cost_rule():
    t = [100.0]
    c = Clock(frame_interval=1.0, kappa=3.0, save_interval=10.0, now=lambda: t[0])
    assert c.frame_due()
    c.frame_done(0.1)                       # max(1, 0.3)
    t[0] += 0.9
    assert not c.frame_due()
    t[0] += 0.2
    assert c.frame_due()
    c.frame_done(2.0)                       # max(1, 6)
    t[0] += 5.0
    assert not c.frame_due()
    t[0] += 1.1
    assert c.frame_due()
    c.frame_done(2.0)
    c.frame_now()
    assert c.frame_due()
    assert not c.save_due()
    t[0] += 10.0
    assert c.save_due()
    c.save_done()
    assert not c.save_due()
    assert not Clock(save_interval=0).save_due()


def test_training_job_runs_to_its_limit_and_lowers_the_loss(tmp_path):
    path, *_ = make_run(tmp_path / "r")
    s = Session("cpu")
    s.open_run(path)
    job = TrainingJob(s, 12)
    first = job.advance()
    while not job.done:
        last = job.advance()
    assert job.count == 12 and last["step"] == 12 and last["done"]
    assert last["loss"] <= first["loss"]
    assert job.status()["limit"] == 12


def test_training_job_without_limit_never_finishes(tmp_path):
    path, *_ = make_run(tmp_path / "r")
    s = Session("cpu")
    s.open_run(path)
    job = TrainingJob(s)
    for _ in range(3):
        job.advance()
    assert not job.done


def test_jobs_need_something_open():
    with pytest.raises(SessionError):
        TrainingJob(Session("cpu"))


def test_seeding_job_adds_atoms_and_stops_at_its_limit(tmp_path):
    s = Session("cpu")
    s.open_dataset(make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False), border_px=5)
    job = SeedingJob(s, trials=30, chunk=10)
    states = []
    while not job.done:
        states.append(job.advance())
    assert job.count == 30 and len(states) == 3
    assert states[-1]["trials"] == 30 and states[-1]["atoms"] == s.info()["layers"][0]["atoms"] > 0
    assert states[-1]["done"] and job.status()["limit"] == 30


def test_seeding_job_notices_a_config_change(tmp_path):
    s = Session("cpu")
    s.open_dataset(make_folder(tmp_path / "f", mode="stem_haadf", with_lattice=False), border_px=5)
    job = SeedingJob(s, trials=30, chunk=5)
    job.advance()
    s.seeding_config({"move_sigma_A": 0.3})        # a live option: the job goes on
    job.advance()
    s.seeding_config({"margin_px": 2.0})           # the seeder has to be rebuilt: the job stops
    with pytest.raises(SessionError, match="changed"):
        job.advance()


def test_seeding_chunk_ends_early_so_stop_is_not_kept_waiting(monkeypatch):
    import time
    from temsim.service import jobs

    class Slow:
        class stats:
            full = False

        def trial(self):
            time.sleep(0.05)

    job = jobs.SeedingJob.__new__(jobs.SeedingJob)
    job.seeder, job.limit, job.chunk, job.tune_every, job.tune_steps = Slow(), None, 1000, 0, 0
    job.count, job._since_tune, job.done = 0, 0, False
    job.session = type("S", (), {"seeder": job.seeder, "seeding_status": lambda self: {}})()
    start = time.monotonic()
    job.advance()
    assert time.monotonic() - start < 1.5 and 0 < job.count < 1000
