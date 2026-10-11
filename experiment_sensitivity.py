
import csv
import math
import random
import statistics
from pathlib import Path

from ant_system import (
    ant_system,
    calculate_length,
    validate_stsp,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
RESULT_DIR = BASE_DIR / "results"

DATASET_ID = "K-40"
N = 40
DATASET_SEED = 1040

SEARCH_SEEDS = [101, 202, 303, 404, 505]

BASE_PARAMS = {
    "m": 10,
    "T": 50,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.5,
    "Q": 100.0,
    "tau0": 1.0,
}

# Sembilan konfigurasi unik.
CONFIGS = [
    ("A-00", 0.0, 2.0),
    ("A-05", 0.5, 2.0),
    ("BASE", 1.0, 2.0),
    ("A-20", 2.0, 2.0),
    ("A-30", 3.0, 2.0),
    ("B-00", 1.0, 0.0),
    ("B-10", 1.0, 1.0),
    ("B-30", 1.0, 3.0),
    ("B-50", 1.0, 5.0),
]


def generate_stsp(n, seed):
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            distance = rng.randint(1, 100)
            D[i][j] = distance
            D[j][i] = distance

    validate_stsp(D)
    return D


def verify_saved_dataset(D):
    path = DATA_DIR / f"{DATASET_ID}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}"
        )

    with path.open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != N:
        raise AssertionError("Ukuran dataset tidak sama.")

    for i, row in enumerate(rows):
        if int(row["city"]) != i:
            raise AssertionError("Indeks kota berbeda.")

        for j in range(N):
            if float(row[str(j)]) != D[i][j]:
                raise AssertionError(
                    f"Jarak berbeda pada ({i}, {j})."
                )


def load_baseline_reference():
    path = RESULT_DIR / "quality_raw.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Referensi tidak ditemukan: {path}"
        )

    reference = {}

    with path.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            if row["dataset"] != DATASET_ID:
                continue

            seed = int(row["search_seed"])

            if seed in reference:
                raise AssertionError(
                    f"Seed referensi duplikat: {seed}"
                )

            reference[seed] = float(
                row["best_length"]
            )

    if set(reference) != set(SEARCH_SEEDS):
        raise AssertionError(
            "Seed referensi K-40 tidak lengkap."
        )

    return reference


def save_csv(path, rows):
    if not rows:
        raise ValueError("Tidak ada data.")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    D = generate_stsp(N, DATASET_SEED)
    verify_saved_dataset(D)

    baseline = load_baseline_reference()

    raw_rows = []
    summary_rows = []

    print("EKSPERIMEN SENSITIVITAS ANT SYSTEM")
    print(f"Dataset: {DATASET_ID}")
    print(f"Seed dataset: {DATASET_SEED}")
    print(f"Konfigurasi unik: {len(CONFIGS)}")
    print("-" * 65)

    for config_id, alpha, beta in CONFIGS:
        params = BASE_PARAMS.copy()
        params["alpha"] = alpha
        params["beta"] = beta

        lengths = []

        for repetition, seed in enumerate(
            SEARCH_SEEDS, start=1
        ):
            result = ant_system(
                D,
                seed=seed,
                **params,
            )

            route = result.best_route

            if sorted(route) != list(range(N)):
                raise AssertionError(
                    f"Rute tidak valid: {config_id}, {seed}"
                )

            verified_length = calculate_length(
                route, D
            )

            if not math.isclose(
                verified_length,
                result.best_length,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise AssertionError(
                    f"Panjang tidak valid: {config_id}"
                )

            # Verifikasi konfigurasi dasar menggunakan
            # hasil aktual pada Subbab 4.5.
            if config_id == "BASE":
                expected = baseline[seed]

                if not math.isclose(
                    result.best_length,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ):
                    raise AssertionError(
                        "Konfigurasi dasar berubah: "
                        f"seed={seed}, "
                        f"hasil={result.best_length}, "
                        f"referensi={expected}"
                    )

            lengths.append(result.best_length)

            raw_rows.append({
                "dataset": DATASET_ID,
                "dataset_seed": DATASET_SEED,
                "config": config_id,
                "alpha": alpha,
                "beta": beta,
                "m": params["m"],
                "T": params["T"],
                "rho": params["rho"],
                "Q": params["Q"],
                "tau0": params["tau0"],
                "repetition": repetition,
                "search_seed": seed,
                "best_length": result.best_length,
                "route": "-".join(map(str, route)),
                "valid": True,
            })

            print(
                f"{config_id:5s} | "
                f"alpha={alpha:.1f} | "
                f"beta={beta:.1f} | "
                f"seed={seed} | "
                f"panjang={result.best_length:.0f}"
            )

        mean = statistics.mean(lengths)
        sd = statistics.stdev(lengths)

        summary_rows.append({
            "dataset": DATASET_ID,
            "config": config_id,
            "alpha": alpha,
            "beta": beta,
            "repetitions": len(lengths),
            "minimum": min(lengths),
            "mean": mean,
            "std_sample": sd,
            "maximum": max(lengths),
        })

        print(
            f"RINGKASAN {config_id}: "
            f"min={min(lengths):.2f}, "
            f"mean={mean:.2f}, "
            f"SD={sd:.2f}, "
            f"max={max(lengths):.2f}"
        )
        print("-" * 65)

    save_csv(
        RESULT_DIR / "sensitivity_raw.csv",
        raw_rows,
    )

    save_csv(
        RESULT_DIR / "sensitivity_summary.csv",
        summary_rows,
    )

    print("EKSPERIMEN SELESAI")
    print(f"Jumlah eksekusi: {len(raw_rows)}")
    print(
        "Data mentah:",
        RESULT_DIR / "sensitivity_raw.csv"
    )
    print(
        "Ringkasan:",
        RESULT_DIR / "sensitivity_summary.csv"
    )


if __name__ == "__main__":
    main()
