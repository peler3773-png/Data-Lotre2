# Data-Lotre2
# 🎯 Prediksi Lotre — Tanpa Jebakan Overdue

Sistem prediksi berbasis LSTM + Optuna yang **TIDAK mengejar angka yang hilang lama**.

> ❌ Dihapus: Pemberat overdue (Makin lama hilang → makin diutamakan)
> ✅ Diganti: Deteksi pola aktif — ikuti yang sedang bergerak, bukan yang "berutang"

---

## 📋 Aturan Logika Baru

| Status | Arti | Tindakan |
|---|---|---|
| 🔥 Panas | Sering muncul belakangan | Utamakan |
| ↩️ Kembali | Baru muncul setelah hilang | Ikuti |
| ⚪ Diam | Belum ada tanda | Tunggu dulu |
| ⛔ Dilarang | Hilang > 30 putaran | Jangan dikejar |

---

## 🚀 Cara Pakai

### 1. Pasang Kebutuhan
```bash
pip install optuna tensorflow
