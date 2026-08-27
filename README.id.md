# Tree Counter

[English](README.md)

Tree Counter adalah plugin QGIS sumber terbuka untuk menghitung pohon pada
citra udara bereferensi geospasial menggunakan model deteksi yang disediakan
pengguna. Kelapa sawit adalah kasus penggunaan pertama yang divalidasi, tetapi
tidak ada bagian plugin yang dikhususkan hanya untuk kelapa sawit.

Seluruh pemrosesan berlangsung di komputer pengguna. Citra dan model tidak
pernah diunggah, tidak ada telemetri, dan jaringan hanya digunakan oleh
Runtime Manager setelah pengguna memulai tindakan instalasi atau pembaruan.

**Rilis ini masih eksperimental dan dalam pengembangan aktif.** Plugin sudah
menjalankan penghitungan dengan model serta raster nyata, tetapi ketepatan
hasil pada data lain tetap bergantung pada model dan citra pengguna.

## Yang dibutuhkan

1. **QGIS 3.44 LTR atau lebih baru**, hingga QGIS 4.x.
2. **Model deteksi milik pengguna.** Tree Counter tidak mengunduh model.
3. **Runtime deteksi**, dipasang satu kali melalui Runtime Manager.

## Instalasi

Untuk release candidate v0.1.0, pasang ZIP yang telah divalidasi melalui
**Plugins → Manage and Install Plugins → Install from ZIP**. Setelah versi
pertama disetujui, paket yang sama juga akan tersedia melalui repositori
plugin resmi QGIS.

Sesudah plugin terbuka, masuk ke **Runtime Manager** lalu tekan **Install**.
Tindakan ini mengunduh ONNX Runtime dan, bila dipilih, PyTorch serta
Ultralytics dari PyPI ke direktori per pengguna di luar instalasi QGIS.
Setiap paket diverifikasi menggunakan versi dan hash yang dipatok sebelum
runtime diaktifkan.

Runtime memerlukan interpreter **Python 3.12** dengan dukungan `venv`, SSL,
dan `ensurepip`. Runtime tersedia untuk Windows x86_64, macOS Apple Silicon,
dan Linux x86_64. Pada Mac Intel, plugin dapat dimuat tetapi runtime v0.1.0
tidak dapat dipasang karena wheel Python 3.12 yang diperlukan tidak tersedia.

## Model

Tree Counter menerima model deteksi **Ultralytics YOLO11**:

| Format | Komponen runtime | Catatan |
| --- | --- | --- |
| `.onnx` | ONNX Runtime | Direkomendasikan; ekspor tanpa NMS bawaan. |
| `.pt` | PyTorch dan Ultralytics | Hanya untuk checkpoint tepercaya. |

Model segmentasi, klasifikasi, pose, keluarga selain YOLO11, layout keluaran
yang tidak dikenal, dan ekspor dengan NMS tertanam akan ditolak. Plugin
menerapkan NMS sendiri agar parameter **NMS IoU** dari UI selalu digunakan.

### Mengapa checkpoint `.pt` perlu dikonfirmasi

Memuat checkpoint PyTorch dapat mengeksekusi kode di dalam berkas tersebut.
Berkas berbahaya dapat melakukan apa pun yang diizinkan akun pengguna.
Karena itu Tree Counter meminta konfirmasi terhadap hash SHA-256 setiap
checkpoint `.pt` baru. Jika isi berkas berubah, konfirmasi harus diulang.

Gunakan `.pt` hanya dari sumber yang dipercaya. Jika memungkinkan, gunakan
ekspor `.onnx` karena graph ONNX merupakan data, bukan checkpoint Python.

## Citra

Tree Counter membaca raster apa pun yang dapat dibuka provider QGIS selama:

- raster bereferensi geospasial dan memiliki CRS yang valid;
- data berupa 8-bit dengan minimal tiga band yang dibaca sebagai RGB.

GeoTIFF, COG, VRT, atau ECW dapat digunakan bila build QGIS/GDAL pengguna
mendukung format tersebut. WMS, WMTS, XYZ, basemap daring, grayscale, 16-bit,
thermal, dan multispektral tidak didukung pada v0.1.0.

Scope dapat berupa seluruh raster, extent peta saat ini, atau layer/fitur
poligon terpilih. Raster diproses per tile pada resolusi aslinya. Overlap tile
akan dideduplikasi secara class-aware agar pohon di batas tile tidak dihitung
dua kali.

## Parameter deteksi

Kontrol utama mencakup Confidence, NMS IoU, Tile size, dan Overlap. Bagian
Advanced menyediakan Duplicate IoU dan Device. NMS IoU berlaku pada deteksi
dalam tile, sedangkan Duplicate IoU digunakan untuk menggabungkan deteksi
lintas tile. Model multikelas menampilkan daftar kelas dan hanya kelas yang
dipilih yang dihitung.

## Keluaran

Hasil ditulis secara atomik ke satu GeoPackage yang berisi:

- `tree_centers` — satu titik untuk setiap pohon;
- `detection_boxes` — kotak deteksi, bila dipilih;
- `run_summary` — model dan hash, parameter, backend, device, jumlah tile,
  durasi, hitungan per kelas, serta peringatan proses.

Layer hasil yang dipilih otomatis ditambahkan ke project QGIS setelah proses
berhasil. Plugin tidak menimpa output secara diam-diam dan tidak menampilkan
hasil parsial sebagai proses sukses. Membatalkan proses akan menghentikan
worker dan tidak menghasilkan file final parsial.

## Device

CPU tersedia pada seluruh profil runtime. Pada Apple Silicon, profil saat ini
dapat menawarkan MPS untuk model `.pt` tepercaya dan CoreML untuk ONNX yang
kompatibel. CUDA hanya akan ditampilkan jika tersedia profil runtime CUDA yang
telah diverifikasi; v0.1.0 belum menyertakannya. Backend dan device yang benar-
benar digunakan dicatat di `run_summary`.

## Keterbatasan

- **Akurasi belum diukur.** Jumlah yang ditampilkan adalah hasil deteksi model,
  bukan klaim jumlah pohon sebenarnya di lapangan.
- Resolusi citra harus sesuai dengan skala yang digunakan saat melatih model.
- Raster besar memerlukan waktu dan ruang sementara yang lebih besar.
- Runtime v0.1.0 tidak tersedia untuk Mac Intel.
- Plugin masih eksperimental dan antarmuka atau skema hasil dapat berubah.

## Pemecahan masalah

**Python 3.12 tidak ditemukan.** Pasang CPython 3.12 lalu ulangi instalasi.
Pada macOS, Tree Counter juga memeriksa lokasi Homebrew dan python.org karena
aplikasi yang dibuka dari Dock menerima `PATH` yang terbatas.

**Model ditolak.** Pesan error menjelaskan penyebabnya: bukan model detect,
bukan YOLO11, memiliki NMS tertanam, layout keluaran tidak dikenal, atau hash
checkpoint belum dipercaya.

**Tidak ada deteksi.** Periksa scope, kecocokan resolusi citra dengan data
latih, kelas yang dipilih, dan nilai confidence.

**Proses gagal.** Buka **Runtime Manager → Open logs**. Log tetap berada di
komputer pengguna dan tidak berisi salinan citra.

## Privasi dan jaringan

- Citra serta model diproses lokal dan tidak pernah diunggah.
- Tidak ada telemetri, analitik, atau unggahan crash otomatis.
- Jaringan hanya digunakan setelah tindakan eksplisit di Runtime Manager.
- Plugin tidak pernah mengunduh model secara otomatis.

## Pengembangan

Lihat [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md),
[SECURITY.md](SECURITY.md), dan [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

```bash
python3 -m pytest -q
python3 -m flake8 tree_counter scripts tests
python3 scripts/check_publication.py
python3 scripts/package_plugin.py
```

## Lisensi

GNU Affero General Public License, hanya versi 3 (AGPL-3.0-only). Lihat
[LICENSE](LICENSE) dan [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
