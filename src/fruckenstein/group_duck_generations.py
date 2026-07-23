"""High-speed generational bake for the relational duck flock.

Each generation begins with several fresh candidate flocks inheriting the same
three body-effect ledgers.  Candidates experience different deterministic gait
phases and replenishment worlds.  Only a lineage that eats, keeps every duck
alive, and remains physically stable may replace the parent inheritance.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import time


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def copy_ledgers(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(1, 4):
        source_path = source / f"duck-{index}.json"
        if source_path.exists():
            shutil.copy2(source_path, destination / source_path.name)


def candidate_lifetime(job: dict) -> dict:
    ledger = Path(job["ledger"]).resolve()
    os.environ["WATCH"] = "0"
    os.environ["GROUP_MEMORY"] = "1"
    os.environ["GROUP_LEDGER_DIR"] = str(ledger)
    os.environ["GROUP_CROWNS"] = "1" if job["crowns"] else "0"
    os.environ["GROUP_CROWN_PERIOD"] = str(job["crown_period"])

    import numpy as np
    import group_ducks as gd

    world = gd.GroupWorld()
    world.rng = np.random.default_rng(job["seed"])
    # A new body need not always wake at the same arbitrary point in its gait
    # or crown rhythm.  These are birth conditions, not mutations to the gait.
    for duck in world.ducks:
        duck.phase = float(
            (job["seed"] * (duck.index + 1) * gd.GOLDEN_ANGLE) % duck.period
        )
        duck.crown_clock = int(
            (job["seed"] + duck.index * 2) % max(1, job["crown_period"])
        )

    minimum_upright = [1.0] * len(world.ducks)
    started = time.time()
    completed_steps = 0
    for completed_steps in range(1, job["steps"] + 1):
        world.step()
        for index, duck in enumerate(world.ducks):
            minimum_upright[index] = min(
                minimum_upright[index], duck.body_state()["upright"]
            )
        if not any(not duck.dead for duck in world.ducks):
            break
    for duck in world.ducks:
        duck.save_memory()

    report = world.report()
    individuals = report["individuals"]
    apples = int(sum(item["apples"] for item in individuals))
    falls = int(sum(item["falls"] for item in individuals))
    deaths = int(len(report["deathEvents"]))
    mean_vitality = float(sum(item["vitality"] for item in individuals) / len(individuals))
    mean_pain = float(sum(item["pain"] for item in individuals) / len(individuals))
    trims = [float(item["bodyMemory"]["steeringTrim"]) for item in individuals]
    enactments = [float(item["bodyMemory"]["gaitEnactment"]) for item in individuals]
    effects = [item["bodyMemory"]["effects"] for item in individuals]
    finite_values = (
        minimum_upright + [mean_vitality, mean_pain] + trims + enactments
        + [float(value) for effect in effects for value in effect.values()]
    )
    finite = all(math.isfinite(value) for value in finite_values)
    gate = {
        "finite": finite,
        "allAlive": report["living"] == len(world.ducks),
        "ate": apples >= 1,
        "noFalls": falls == 0,
        "upright": min(minimum_upright) >= 0.85,
        "painBounded": mean_pain < 0.45,
        "gaitBounded": all(0.36 <= value <= 0.44 for value in enactments),
        "trimBounded": all(
            gd.GROUP_STEERING_RANGE[0] <= value <= gd.GROUP_STEERING_RANGE[1]
            for value in trims
        ),
    }
    accepted = all(gate.values())
    # Apples dominate selection.  Survival, vitality, and bodily stability
    # break ties; falls and deaths cannot be compensated for by food.
    score = (
        120.0 * apples
        + 30.0 * report["living"]
        + 20.0 * mean_vitality
        + 10.0 * min(minimum_upright)
        - 80.0 * falls
        - 160.0 * deaths
        - 15.0 * mean_pain
    )
    return {
        "candidate": job["candidate"],
        "seed": job["seed"],
        "ledger": str(ledger),
        "steps": completed_steps,
        "elapsedSeconds": time.time() - started,
        "accepted": accepted,
        "gate": gate,
        "score": score,
        "living": report["living"],
        "apples": apples,
        "falls": falls,
        "deaths": deaths,
        "contacts": report["contactEvents"],
        "knockdowns": len(report["knockdownEvents"]),
        "minimumUpright": minimum_upright,
        "meanVitality": mean_vitality,
        "meanPain": mean_pain,
        "trims": trims,
        "gaitEnactments": enactments,
        "effects": effects,
        "bodyMemory": [item["bodyMemory"] for item in individuals],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--crown-period", type=int, default=5)
    parser.add_argument("--no-crowns", action="store_true")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    ledger = args.ledger.resolve()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "bake-report.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    initial = run_dir / "initial-inheritance"
    copy_ledgers(ledger, initial)
    started = time.time()
    payload = {
        "status": "checkpoint-incomplete",
        "definition": (
            "fresh bodies; shared parent inheritance; candidate selection by "
            "food, survival, and physical stability"
        ),
        "ledger": str(ledger),
        "generationsRequested": args.generations,
        "candidatesPerGeneration": args.candidates,
        "stepsPerLifetime": args.steps,
        "crowns": not args.no_crowns,
        "crownPeriod": args.crown_period,
        "generations": [],
    }
    atomic_json(report_path, payload)

    for generation in range(1, args.generations + 1):
        generation_dir = run_dir / f"generation-{generation:02d}"
        parent = generation_dir / "parent"
        copy_ledgers(ledger, parent)
        jobs = []
        for candidate in range(1, args.candidates + 1):
            candidate_dir = generation_dir / f"candidate-{candidate:02d}"
            copy_ledgers(parent, candidate_dir)
            jobs.append({
                "generation": generation,
                "candidate": candidate,
                "seed": args.seed + generation * 7919 + candidate * 104729,
                "ledger": str(candidate_dir),
                "steps": args.steps,
                "crowns": not args.no_crowns,
                "crown_period": args.crown_period,
            })

        workers = min(args.candidates, max(1, os.cpu_count() or 1))
        candidates = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(candidate_lifetime, job) for job in jobs]
            for future in as_completed(futures):
                candidates.append(future.result())
        candidates.sort(key=lambda item: item["candidate"])
        eligible = [item for item in candidates if item["accepted"]]
        winner = max(eligible, key=lambda item: item["score"]) if eligible else None
        if winner is not None:
            selected = Path(winner["ledger"])
            incoming = generation_dir / "selected-incoming"
            copy_ledgers(selected, incoming)
            copy_ledgers(incoming, ledger)

        generation_report = {
            "generation": generation,
            "promoted": winner is not None,
            "winner": winner["candidate"] if winner is not None else None,
            "winnerScore": winner["score"] if winner is not None else None,
            "candidates": candidates,
        }
        payload["generations"].append(generation_report)
        payload["elapsedSeconds"] = time.time() - started
        atomic_json(report_path, payload)

    payload["status"] = "checkpoint-complete"
    payload["elapsedSeconds"] = time.time() - started
    payload["summary"] = {
        "generations": args.generations,
        "promotions": sum(item["promoted"] for item in payload["generations"]),
        "candidateLifetimes": args.generations * args.candidates,
        "duckLifetimes": args.generations * args.candidates * 3,
        "selectedApples": sum(
            next(
                candidate["apples"]
                for candidate in generation["candidates"]
                if candidate["candidate"] == generation["winner"]
            )
            for generation in payload["generations"]
            if generation["winner"] is not None
        ),
        "selectedFalls": sum(
            next(
                candidate["falls"]
                for candidate in generation["candidates"]
                if candidate["candidate"] == generation["winner"]
            )
            for generation in payload["generations"]
            if generation["winner"] is not None
        ),
    }
    atomic_json(report_path, payload)


if __name__ == "__main__":
    main()
