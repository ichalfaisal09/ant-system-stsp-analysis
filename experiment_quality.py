
import csv
import random
import statistics
from pathlib import Path

from ant_system import ant_system, calculate_length, validate_stsp


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "datasets"
RESULT_DIR = BASE / "results"

SIZES = [10, 20, 40, 60, 80]
SEARCH_SEEDS = [101, 202, 303, 404, 505]

PARAMS = {
    "m": 10,
    "T": 50,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.5,
    "Q": 100.0,
    "tau0": 1.0,
}


def generate_stsp(n, seed):
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            d = rng.randint(1, 100)
            D[i][j] = d
            D[j][i] = d

    validate_stsp(D)
    return D


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    raw_rows = []
    summary_rows = []

    for n in SIZES:
        dataset_seed = 1000 + n
        dataset_id = f"K-{n:02d}"

        D = generate_stsp(n, dataset_seed)

        write_csv(
            DATA_DIR / f"{dataset_id}.csv",
            [
                {"city": i, **{str(j): D[i][j] for j in range(n)}}
                for i in range(n)
            ],
            ["city"] + [str(j) for j in range(n)],
        )

        lengths = []

        for repetition, seed in enumerate(SEARCH_SEEDS, 1):
            result = ant_system(D, seed=seed, **PARAMS)

            route = result.best_route

            if len(route) != n or sorted(route) != list(range(n)):
                raise AssertionError(
                    f"Rute tidak valid: {dataset_id}, seed={seed}"
                )

            recalculated = calculate_length(route, D)

            if abs(recalculated - result.best_length) > 1e-9:
                raise AssertionError(
                    f"Panjang tidak konsisten: {dataset_id}"
                )

            lengths.append(result.best_length)

            raw_rows.append({
                "dataset": dataset_id,
                "n": n,
                "dataset_seed": dataset_seed,
                "repetition": repetition,
                "search_seed": seed,
                "m": PARAMS["m"],
                "T": PARAMS["T"],
                "alpha": PARAMS["alpha"],
                "beta": PARAMS["beta"],
                "rho": PARAMS["rho"],
                "Q": PARAMS["Q"],
                "tau0": PARAMS["tau0"],
                "best_length": result.best_length,
                "route": "-".join(map(str, route)),
                "valid": True,
            })

        summary_rows.append({
            "dataset": dataset_id,
            "n": n,
            "repetitions": len(lengths),
            "minimum": min(lengths),
            "mean": statistics.mean(lengths),
            "std_sample": statistics.stdev(lengths),
            "maximum": max(lengths),
        })

        print(
            f"{dataset_id}: "
            f"minimum={min(lengths):.2f}, "
            f"rata-rata={statistics.mean(lengths):.2f}, "
            f"SD={statistics.stdev(lengths):.2f}"
        )

    write_csv(
        RESULT_DIR / "quality_raw.csv",
        raw_rows,
        list(raw_rows[0].keys()),
    )

    write_csv(
        RESULT_DIR / "quality_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )

    print("\nEksperimen kualitas solusi selesai.")
    print(f"Data mentah: {RESULT_DIR / 'quality_raw.csv'}")
    print(f"Ringkasan: {RESULT_DIR / 'quality_summary.csv'}")


if __name__ == "__main__":
    main()
