
import csv
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

DATASETS = ["K-10", "K-20", "K-40", "K-60", "K-80"]
SIZES = [10, 20, 40, 60, 80]
SEEDS = {101, 202, 303, 404, 505}


def read_csv(filename):
    path = RESULTS / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    with path.open(
        newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"CSV kosong: {filename}")

    return rows


def valid_number(value, context):
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"Nilai tidak berhingga: {context}"
        )

    return result


def order_by_dataset(rows, field="dataset"):
    lookup = {}

    for row in rows:
        key = row[field]

        if key in lookup:
            raise ValueError(
                f"Dataset duplikat: {key}"
            )

        lookup[key] = row

    if set(lookup) != set(DATASETS):
        raise ValueError(
            "Daftar dataset tidak sesuai: "
            f"{sorted(lookup)}"
        )

    return [lookup[key] for key in DATASETS]


def save_figure(filename):
    FIGURES.mkdir(parents=True, exist_ok=True)

    path = FIGURES / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close()
    print(f"Disimpan: {path}")


def configure_plot():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def figure_runtime():
    """
    Gambar 4.1.

    Kurva referensi c*n^2 dinormalisasi
    menggunakan hasil waktu K-80.

    Grafik tidak menyatakan waktu teoritis absolut.
    """
    rows = order_by_dataset(
        read_csv("runtime_summary.csv")
    )

    n = np.array(
        [int(row["n"]) for row in rows],
        dtype=float,
    )

    means = np.array([
        valid_number(row["mean_seconds"], "waktu")
        for row in rows
    ])

    stds = np.array([
        valid_number(row["std_seconds"], "SD waktu")
        for row in rows
    ])

    if not np.array_equal(
        n, np.array(SIZES, dtype=float)
    ):
        raise ValueError("Ukuran runtime tidak sesuai.")

    if np.any(means <= 0) or np.any(stds < 0):
        raise ValueError("Statistik runtime tidak valid.")

    # Tetapkan kurva referensi di titik K-80.
    c = means[-1] / n[-1] ** 2
    theoretical = c * n ** 2

    plt.figure(figsize=(8.5, 5.2))

    plt.errorbar(
        n,
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
        linewidth=2,
        label="Waktu empiris (rata-rata ± SD)",
    )

    plt.plot(
        n,
        theoretical,
        "--",
        linewidth=2,
        label="Referensi kuadratik c·n²",
    )

    plt.xlabel("Jumlah kota (n)")
    plt.ylabel("Waktu eksekusi (detik)")
    plt.title(
        "Waktu Eksekusi Ant System dan "
        "Referensi Kompleksitas Kuadratik"
    )
    plt.xticks(n.astype(int))
    plt.grid(alpha=0.25)
    plt.legend()

    plt.figtext(
        0.5,
        0.01,
        "Kurva referensi dinormalisasi pada K-80; "
        "m = 10 dan T = 50 tetap.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    save_figure("gambar_4_1_runtime_vs_theory.png")


def load_quality():
    """
    Membaca data mentah kualitas AS.

    Memastikan tiap dataset mempunyai lima seed
    tanpa duplikasi dan panjang rute positif.
    """
    rows = read_csv("quality_raw.csv")
    grouped = {dataset: {} for dataset in DATASETS}

    for row in rows:
        dataset = row["dataset"]

        if dataset not in grouped:
            raise ValueError(
                f"Dataset tidak dikenal: {dataset}"
            )

        seed = int(row["search_seed"])
        length = valid_number(
            row["best_length"], "panjang rute"
        )

        if seed in grouped[dataset]:
            raise ValueError(
                f"Seed duplikat: {dataset}, {seed}"
            )

        if length <= 0:
            raise ValueError("Panjang tidak positif.")

        grouped[dataset][seed] = length

    means = []
    stds = []

    for dataset in DATASETS:
        if set(grouped[dataset]) != SEEDS:
            raise ValueError(
                f"Seed tidak lengkap: {dataset}"
            )

        values = [
            grouped[dataset][seed]
            for seed in sorted(SEEDS)
        ]

        means.append(statistics.mean(values))
        stds.append(statistics.stdev(values))

    return np.array(means), np.array(stds)


def figure_quality():
    """
    Gambar 4.2.
    Rata-rata dan SD panjang rute AS.

    Tidak menafsirkan tren antarinstance
    independen sebagai perubahan kualitas
    akibat ukuran kota.
    """
    means, stds = load_quality()
    x = np.arange(len(DATASETS))

    plt.figure(figsize=(8.5, 5.2))

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.85,
    )

    plt.xticks(x, DATASETS)
    plt.ylabel("Panjang rute terbaik")
    plt.xlabel("Instance STSP")
    plt.title(
        "Kualitas Solusi Ant System "
        "(Lima Seed per Instance)"
    )
    plt.grid(axis="y", alpha=0.25)

    for i, value in enumerate(means):
        plt.text(
            i,
            value + stds[i] + 7,
            f"{value:.1f}",
            ha="center",
            fontsize=9,
        )

    plt.ylim(
        0,
        max(means + stds) * 1.22,
    )

    plt.figtext(
        0.5,
        0.01,
        "Setiap instance menggunakan matriks jarak "
        "independen; nilai bukan gap optimum.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    save_figure("gambar_4_2_solution_quality.png")


def figure_sensitivity():
    """
    Gambar 4.3.
    Dua panel dalam satu gambar:
    sensitivitas alpha dan beta.
    """
    rows = read_csv("sensitivity_summary.csv")
    by_config = {}

    for row in rows:
        key = row["config"]

        if key in by_config:
            raise ValueError(
                f"Konfigurasi duplikat: {key}"
            )

        by_config[key] = row

    expected = {
        "A-00", "A-05", "BASE",
        "A-20", "A-30",
        "B-00", "B-10", "B-30", "B-50",
    }

    if set(by_config) != expected:
        raise ValueError(
            "Konfigurasi sensitivitas tidak lengkap."
        )

    alpha_codes = [
        "A-00", "A-05", "BASE",
        "A-20", "A-30",
    ]

    beta_codes = [
        "B-00", "B-10", "BASE",
        "B-30", "B-50",
    ]

    def values(codes, parameter):
        x = []
        means = []
        stds = []

        for code in codes:
            row = by_config[code]

            if int(row["repetitions"]) != 5:
                raise ValueError(
                    f"Repetisi tidak sesuai: {code}"
                )

            x.append(float(row[parameter]))
            means.append(float(row["mean"]))
            stds.append(float(row["std_sample"]))

        return x, means, stds

    ax, ay, asd = values(alpha_codes, "alpha")
    bx, by, bsd = values(beta_codes, "beta")

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8)
    )

    axes[0].errorbar(
        ax, ay, yerr=asd,
        fmt="o-", linewidth=2, capsize=4,
    )
    axes[0].set_title(
        "Pengaruh α (β = 2)"
    )
    axes[0].set_xlabel("Nilai α")
    axes[0].set_ylabel(
        "Rata-rata panjang rute"
    )
    axes[0].grid(alpha=0.25)
    axes[0].set_xticks(ax)

    axes[1].errorbar(
        bx, by, yerr=bsd,
        fmt="o-", linewidth=2, capsize=4,
    )
    axes[1].set_title(
        "Pengaruh β (α = 1)"
    )
    axes[1].set_xlabel("Nilai β")
    axes[1].set_ylabel(
        "Rata-rata panjang rute"
    )
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(bx)

    fig.suptitle(
        "Analisis Sensitivitas Parameter Ant System",
        fontsize=13,
    )

    fig.text(
        0.5,
        0.01,
        "Instance K-40; lima seed per konfigurasi. "
        "Garis galat menunjukkan SD sampel.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    save_figure("gambar_4_3_parameter_sensitivity.png")


def figure_comparison():
    """
    Gambar 4.4.
    Perbandingan kualitas solusi AS dan NN.
    """
    rows = order_by_dataset(
        read_csv("comparison_summary.csv")
    )

    as_means = np.array([
        float(row["as_mean_length"])
        for row in rows
    ])

    as_stds = np.array([
        float(row["as_std_length"])
        for row in rows
    ])

    nn_lengths = np.array([
        float(row["nn_length"])
        for row in rows
    ])

    if (
        np.any(as_means <= 0)
        or np.any(as_stds < 0)
        or np.any(nn_lengths <= 0)
    ):
        raise ValueError(
            "Data kualitas perbandingan tidak sah."
        )

    # Cocokkan data AS dengan kualitas utama.
    quality_means, quality_stds = load_quality()

    if not np.allclose(as_means, quality_means):
        raise AssertionError(
            "Rata-rata AS berbeda dari quality_raw.csv."
        )

    if not np.allclose(as_stds, quality_stds):
        raise AssertionError(
            "SD AS berbeda dari quality_raw.csv."
        )

    x = np.arange(len(DATASETS))
    width = 0.36

    plt.figure(figsize=(9, 5.2))

    plt.bar(
        x - width / 2,
        as_means,
        width,
        yerr=as_stds,
        capsize=4,
        label="Ant System (rata-rata ± SD)",
    )

    plt.bar(
        x + width / 2,
        nn_lengths,
        width,
        label="Nearest Neighbor",
    )

    plt.xticks(x, DATASETS)
    plt.xlabel("Instance STSP")
    plt.ylabel("Panjang rute")
    plt.title(
        "Perbandingan Kualitas Solusi "
        "Ant System dan Nearest Neighbor"
    )
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.ylim(
        0,
        max(
            max(as_means + as_stds),
            max(nn_lengths),
        ) * 1.20,
    )

    plt.figtext(
        0.5,
        0.01,
        "AS: 500 konstruksi per eksekusi; "
        "NN: satu konstruksi dari kota 0.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    save_figure("gambar_4_4_as_vs_nn.png")


def main():
    configure_plot()

    figure_runtime()
    figure_quality()
    figure_sensitivity()
    figure_comparison()

    print("-" * 60)
    print("SELESAI: 4 grafik berhasil dibuat.")
    print(f"Direktori grafik: {FIGURES}")


if __name__ == "__main__":
    main()
