# Penelitian Ant System–Ant-Cycle untuk STSP

## 1. Deskripsi

Proyek ini merupakan implementasi dan evaluasi algoritma
Ant System (AS) dengan mekanisme Ant-Cycle untuk menyelesaikan
Symmetric Traveling Salesman Problem (STSP).

Penelitian mencakup:

- Pengujian kebenaran implementasi.
- Analisis kualitas solusi.
- Pengukuran waktu eksekusi.
- Analisis sensitivitas parameter alpha dan beta.
- Perbandingan dengan Nearest Neighbor (NN).
- Pengukuran gap optimalitas pada instance kecil.

Algoritma inti diimplementasikan secara mandiri menggunakan Python.

## 2. Lingkungan Penelitian

Lingkungan eksperimen asli:

- Sistem operasi: Windows 11
- Python: 3.10.11
- Pengujian otomatis: pytest
- Visualisasi: Matplotlib
- Pengolahan numerik: NumPy

Waktu eksekusi dapat berbeda pada perangkat lain.

## 3. Struktur Proyek

```text

Penelitian-ACO/
├── ant\_system.py
├── nearest\_neighbor.py
├── test\_ant\_system.py
├── test\_nearest\_neighbor.py
├── experiment\_quality.py
├── experiment\_runtime.py
├── experiment\_sensitivity.py
├── experiment\_comparison.py
├── experiment\_optimality\_gap.py
├── benchmark\_nn.py
├── generate\_figures.py
├── requirements.txt
├── README.md
├── datasets/
├── results/
└── figures/
```

## 4. Persiapan Lingkungan

Buka PowerShell pada direktori proyek.
Buat virtual environment:

```powershell
python -m venv .venv
```

Instal dependensi:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

## 5. Dataset

Dataset utama terdiri atas lima instance STSP:
| Instance | Kota | Seed dataset |
|---|---:|---:|
| K-10 | 10 | 1010 |
| K-20 | 20 | 1020 |
| K-40 | 40 | 1040 |
| K-60 | 60 | 1060 |
| K-80 | 80 | 1080 |

Jarak antarkota dibangkitkan secara simetris menggunakan

`random.Random(1000 + n)` dan `randint(1, 100)`.

Instance validasi kecil:

- V-03: optimum global 9.
- V-04: optimum global 13.
- V-05: optimum global 18.

Instance kecil didefinisikan dalam kode pengujian dan eksperimen gap optimalitas.

## 6. Parameter Dasar Ant System

| Parameter          | Nilai |
| ------------------ | ----: |
| Jumlah semut (m)   |    10 |
| Jumlah iterasi (T) |    50 |
| Alpha              |   1.0 |
| Beta               |   2.0 |
| Rho                |   0.5 |
| Q                  | 100.0 |
| Tau awal           |   1.0 |

Seed pencarian:

`101, 202, 303, 404, 505`

Setiap eksekusi menghasilkan 500 konstruksi rute.

## 7. Pengujian Kebenaran

Jalankan:

```powershell

.\\.venv\\Scripts\\python.exe -m pytest test\_ant\_system.py test\_nearest\_neighbor.py -v

```

Hasil referensi penelitian:

`28 passed`

Pengujian meliputi validasi STSP, evaluasi rute, reproduksibilitas, pembaruan Ant-Cycle, dan NN.

## 8. Menjalankan Eksperimen

Jalankan perintah berikut secara berurutan:

### 8.1 Kualitas Solusi

```powershell
.\\.venv\\Scripts\\python.exe experiment\_quality.py
```

### 8.2 Waktu Eksekusi

```powershell
.\\.venv\\Scripts\\python.exe experiment\_runtime.py
```

### 8.3 Sensitivitas Parameter

```powershell
.\\.venv\\Scripts\\python.exe experiment\_sensitivity.py
```

### 8.4 Perbandingan Ant System dan NN

```powershell
.\\.venv\\Scripts\\python.exe experiment\_comparison.py
```

### 8.5 Benchmark Nearest Neighbor

```powershell
.\\.venv\\Scripts\\python.exe benchmark\_nn.py
```

### 8.6 Gap Optimalitas

```powershell
.\\.venv\\Scripts\\python.exe experiment\_optimality\_gap.py
```

### 8.7 Pembuatan Grafik

```powershell
.\\.venv\\Scripts\\python.exe generate\_figures.py
```

## 9. Hasil Referensi

Rata-rata panjang rute Ant System:
| Instance | Rata-rata |
|---|---:|
| K-10 | 222.0 |
| K-20 | 191.0 |
| K-40 | 414.4 |
| K-60 | 258.2 |
| K-80 | 343.2 |

Seluruh 15 pengujian gap optimalitas pada instance kecil mencapai optimum global dengan gap 0%.

Data rinci tersedia dalam direktori `results/`.

## 10. Reproduksibilitas

Hasil panjang rute pada dataset, parameter, dan seed yang sama diharapkan dapat direproduksi menggunakan implementasi dan lingkungan Python yang kompatibel.

Waktu eksekusi dipengaruhi oleh perangkat dan kondisi sistem.

PERHATIAN:

Program eksperimen dapat menimpa CSV dalam `results/`.
Untuk mempertahankan data referensi penelitian, jalankan pengulangan pada salinan folder proyek.

## 11. Kompleksitas

Berdasarkan analisis operasi algoritma:

- Waktu: Theta(T \* m \* n^2)
- Ruang: Theta(n^2 + m \* n)

Algoritma tidak menjamin penemuan optimum global pada semua instance dalam jumlah iterasi terbatas.
