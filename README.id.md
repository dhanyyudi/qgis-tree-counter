# Tree Counter

Tree Counter adalah plugin QGIS sumber terbuka untuk menghitung pohon pada citra udara bereferensi geospasial dengan model deteksi Ultralytics YOLO yang disediakan pengguna. Penghitungan kelapa sawit merupakan kasus penggunaan tervalidasi pertama, tetapi proyek ini ditujukan untuk penghitungan pohon secara umum.

Proyek ini sedang dalam pengembangan aktif dan belum siap dipasang untuk produksi. Repositori saat ini berisi fondasi plugin publik; inferensi dan alur penghitungan terfokus direncanakan untuk rilis mendatang.

## Alur yang direncanakan

Plugin dirancang untuk menerima model deteksi YOLO lokal berformat `.pt` dan `.onnx`, membaca layer raster yang didukung melalui QGIS, menjalankan inferensi secara lokal dalam runtime terisolasi setelah pemasangan, dan menghasilkan keluaran GeoPackage bereferensi geospasial. Keluaran yang direncanakan mencakup titik pusat pohon, kotak deteksi, dan provenance proses. Model serta citra tetap berada di komputer pengguna.

Runtime akan dipasang dan dikelola secara eksplisit oleh pengguna. Plugin tidak akan menyertakan berkas model, citra raster, binary runtime, atau wheel Python, dan tidak akan mengunduh model secara otomatis. Plugin tidak akan mengirim model atau citra raster ke layanan jarak jauh, serta tidak akan mengumpulkan telemetri atau analitik maupun mengunggah laporan crash secara otomatis. Akses jaringan hanya akan terjadi setelah pengguna secara eksplisit memulai tindakan di Runtime Manager.

## Kompatibilitas

Targetnya adalah satu paket untuk QGIS 3.44 LTR hingga QGIS 4.x (maksimal 4.99) pada Windows, macOS, dan Ubuntu LTS. CPU adalah baseline wajib; akselerasi perangkat keras yang kompatibel bersifat opsional.

## Status pengembangan

Lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk ekspektasi pengembangan dan [CHANGELOG.md](CHANGELOG.md) untuk riwayat rilis. Proyek ini dilisensikan berdasarkan GNU Affero General Public License, hanya versi 3 (AGPL-3.0-only).
