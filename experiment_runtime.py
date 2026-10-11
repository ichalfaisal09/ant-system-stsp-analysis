
import csv
import math
import random
import statistics
import time
from pathlib import Path

from ant_system import ant_system, calculate_length, validate_stsp


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
RESULT_DIR = BASE_DIR / "results"

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
    """Bangkitkan matriks yang sama dengan eksperimen 4.5."""
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            distance = rng.randint(1, 100)
            D[i][j] = distance
            D[j][i] = distance

    validate_stsp(D)
    return D


def verify_dataset(dataset_id, D):
    """Pastikan matriks sesuai berkas dataset 4.5."""
    path = DATA_DIR / f"{dataset_id}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}"
        )

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(D)

    if len(rows) != n:
        raise AssertionError(
            f"Ukuran dataset berbeda: {dataset_id}"
        )

    for i, row in enumerate(rows):
        if int(row["city"]) != i:
            raise AssertionError("Indeks kota tidak sesuai.")

        for j in range(n):
            if float(row[str(j)]) != D[i][j]:
                raise AssertionError(
                    f"Bobot berbeda pada {dataset_id}: "
                    f"({i}, {j})"
                )


def load_quality_reference():
    """Ambil hasil kualitas aktual pada Subbab 4.5."""
    path = RESULT_DIR / "quality_raw.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Referensi kualitas tidak ditemukan: {path}"
        )

    reference = {}

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], int(row["search_seed"]))

            if key in reference:
                raise AssertionError(
                    f"Referensi kualitas duplikat: {key}"
                )

            reference[key] = float(row["best_length"])

    return reference


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    reference = load_quality_reference()

    raw = []
    summary = []

    print("EKSPERIMEN WAKTU EKSEKUSI ANT SYSTEM")
    print("Parameter:", PARAMS)
    print("-" * 55)

    for n in SIZES:
        dataset_id = f"K-{n:02d}"
        dataset_seed = 1000 + n

        # Seluruh penyiapan dataset di luar timer.
        D = generate_stsp(n, dataset_seed)
        verify_dataset(dataset_id, D)

        # Pemanasan interpreter untuk ukuran ini.
        # Hasil dan waktunya tidak dicatat.
        ant_system(D, seed=9999, **PARAMS)

        times = []

        for repetition, seed in enumerate(SEARCH_SEEDS, 1):

            # Pengukuran hanya membungkus pemanggilan
            # ant_system (termasuk validasi internalnya).
            start = time.perf_counter()
            result = ant_system(D, seed=seed, **PARAMS)
            elapsed = time.perf_counter() - start

            # Semua pemeriksaan di luar interval timer.
            route = result.best_route

            if sorted(route) != list(range(n)):
                raise AssertionError(
                    f"Rute tidak valid: {dataset_id}, {seed}"
                )

            length = calculate_length(route, D)

            if not math.isclose(
                length,
                result.best_length,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise AssertionError(
                    "Panjang rute tidak konsisten."
                )

            key = (dataset_id, seed)

            if key not in reference:
                raise AssertionError(
                    f"Referensi kualitas hilang: {key}"
                )

            if not math.isclose(
                result.best_length,
                reference[key],
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise AssertionError(
                    "Hasil kualitas berubah dibanding 4.5: "
                    f"{dataset_id}, seed={seed}"
                )

            times.append(elapsed)

            raw.append({
                "dataset": dataset_id,
                "n": n,
                "dataset_seed": dataset_seed,
                "repetition": repetition,
                "search_seed": seed,
                **PARAMS,
                "time_seconds": elapsed,
                "best_length": result.best_length,
                "quality_verified": True,
            })

            print(
                f"{dataset_id} | seed={seed} | "
                f"waktu={elapsed:.6f} detik | "
                f"panjang={result.best_length:.0f}"
            )

        mean = statistics.mean(times)
        std = statistics.stdev(times)

        summary.append({
            "dataset": dataset_id,
            "n": n,
            "repetitions": len(times),
            "minimum_seconds": min(times),
            "mean_seconds": mean,
            "std_seconds": std,
            "maximum_seconds": max(times),
            "mean_over_n_squared": mean / (n * n),
        })

        print(
            f"Ringkasan {dataset_id}: "
            f"mean={mean:.6f}s, "
            f"SD={std:.6f}s, "
            f"t/n²={mean / (n * n):.9f}"
        )
        print("-" * 55)

    save_csv(
        RESULT_DIR / "runtime_raw.csv",
        raw,
        list(raw[0].keys()),
    )

    save_csv(
        RESULT_DIR / "runtime_summary.csv",
        summary,
        list(summary[0].keys()),
    )

    print("EKSPERIMEN SELESAI")
    print("Total pengukuran:", len(raw))
    print("Data mentah:", RESULT_DIR / "runtime_raw.csv")
    print("Ringkasan:", RESULT_DIR / "runtime_summary.csv")


if __name__ == "__main__":
    main()
