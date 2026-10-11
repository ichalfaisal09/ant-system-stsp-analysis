
import csv
import math
import statistics
import time
from pathlib import Path

from ant_system import ant_system, calculate_length, validate_stsp
from nearest_neighbor import nearest_neighbor


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "datasets"
RESULT_DIR = BASE / "results"

SIZES = [10, 20, 40, 60, 80]
SEEDS = [101, 202, 303, 404, 505]

PARAMS = {
    "m": 10,
    "T": 50,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.5,
    "Q": 100.0,
    "tau0": 1.0,
}


def read_dataset(n):
    path = DATA_DIR / f"K-{n:02d}.csv"

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != n:
        raise AssertionError("Ukuran dataset tidak sesuai.")

    D = []

    for i, row in enumerate(rows):
        if int(row["city"]) != i:
            raise AssertionError("Indeks kota tidak sesuai.")

        D.append([
            float(row[str(j)])
            for j in range(n)
        ])

    validate_stsp(D)
    return D


def read_quality_reference():
    path = RESULT_DIR / "quality_raw.csv"
    reference = {}

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["dataset"],
                int(row["search_seed"]),
            )

            if key in reference:
                raise AssertionError("Referensi duplikat.")

            reference[key] = float(row["best_length"])

    return reference


def check_result(route, length, D):
    n = len(D)

    if sorted(route) != list(range(n)):
        raise AssertionError("Rute tidak valid.")

    expected = calculate_length(route, D)

    if not math.isclose(
        expected, length,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise AssertionError("Panjang rute tidak konsisten.")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    quality_reference = read_quality_reference()

    raw = []
    summary = []

    print("PERBANDINGAN ANT SYSTEM DAN NEAREST NEIGHBOR")
    print("-" * 65)

    for n in SIZES:
        dataset = f"K-{n:02d}"
        D = read_dataset(n)

        # Pemanasan tidak dicatat.
        ant_system(D, seed=9999, **PARAMS)
        nearest_neighbor(D)

        as_lengths = []
        as_times = []
        nn_lengths = []
        nn_times = []

        for repetition, seed in enumerate(SEEDS, 1):
            # Pengukuran AS.
            start = time.perf_counter()
            as_result = ant_system(
                D, seed=seed, **PARAMS
            )
            as_elapsed = time.perf_counter() - start

            # Pengukuran NN pada sesi yang sama.
            start = time.perf_counter()
            nn_result = nearest_neighbor(D)
            nn_elapsed = time.perf_counter() - start

            # Pemeriksaan dilakukan di luar timer.
            check_result(
                as_result.best_route,
                as_result.best_length,
                D,
            )

            check_result(
                nn_result.route,
                nn_result.length,
                D,
            )

            key = (dataset, seed)

            if key not in quality_reference:
                raise AssertionError(
                    f"Referensi tidak ditemukan: {key}"
                )

            if not math.isclose(
                as_result.best_length,
                quality_reference[key],
                abs_tol=1e-9,
                rel_tol=1e-12,
            ):
                raise AssertionError(
                    f"Hasil AS berbeda: {dataset}, seed={seed}"
                )

            as_lengths.append(as_result.best_length)
            as_times.append(as_elapsed)
            nn_lengths.append(nn_result.length)
            nn_times.append(nn_elapsed)

            raw.append({
                "dataset": dataset,
                "n": n,
                "repetition": repetition,
                "search_seed": seed,
                "as_length": as_result.best_length,
                "nn_length": nn_result.length,
                "as_seconds": as_elapsed,
                "nn_seconds": nn_elapsed,
                "as_route": "-".join(
                    map(str, as_result.best_route)
                ),
                "nn_route": "-".join(
                    map(str, nn_result.route)
                ),
            })

            print(
                f"{dataset} | seed={seed} | "
                f"AS={as_result.best_length:.0f} | "
                f"NN={nn_result.length:.0f} | "
                f"tAS={as_elapsed:.6f}s | "
                f"tNN={nn_elapsed:.6f}s"
            )

        as_mean_length = statistics.mean(as_lengths)
        nn_length = nn_lengths[0]
        as_mean_time = statistics.mean(as_times)
        nn_mean_time = statistics.mean(nn_times)

        if any(x != nn_length for x in nn_lengths):
            raise AssertionError(
                "NN berubah pada dataset yang sama."
            )

        quality_difference = (
            (as_mean_length - nn_length)
            / nn_length * 100.0
        )

        time_ratio = as_mean_time / nn_mean_time

        summary.append({
            "dataset": dataset,
            "n": n,
            "as_mean_length": as_mean_length,
            "as_min_length": min(as_lengths),
            "as_std_length": statistics.stdev(as_lengths),
            "nn_length": nn_length,
            "quality_difference_pct": quality_difference,
            "as_mean_seconds": as_mean_time,
            "as_std_seconds": statistics.stdev(as_times),
            "nn_mean_seconds": nn_mean_time,
            "nn_std_seconds": statistics.stdev(nn_times),
            "time_ratio": time_ratio,
        })

        print(
            f"RINGKASAN {dataset}: "
            f"AS mean={as_mean_length:.2f}; "
            f"NN={nn_length:.2f}; "
            f"selisih={quality_difference:+.2f}%; "
            f"rasio waktu={time_ratio:.2f}x"
        )
        print("-" * 65)

    write_csv(
        RESULT_DIR / "comparison_raw.csv", raw
    )

    write_csv(
        RESULT_DIR / "comparison_summary.csv", summary
    )

    print("EKSPERIMEN SELESAI")
    print("Jumlah pasangan pengukuran:", len(raw))
    print(
        "Ringkasan:",
        RESULT_DIR / "comparison_summary.csv"
    )


if __name__ == "__main__":
    main()
