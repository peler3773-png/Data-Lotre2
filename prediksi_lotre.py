import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED_TETAP = 20261005
import random
random.seed(SEED_TETAP)
import numpy as np
np.random.seed(SEED_TETAP)
import tensorflow as tf
tf.random.set_seed(SEED_TETAP)
tf.get_logger().setLevel('ERROR')
from collections import Counter

try:
    import optuna
except ImportError:
    print("ERROR: Pasang dulu: pip install optuna")
    sys.exit(1)

import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 12
LIMIT_PER_PASARAN = 2000
AMBANG_KEMBALI = 2
JENDELA_PANAS = 15
BAHAYA_TERLALU_LAMA = 30

OPTUNA_EPOCH_MIN = 49
OPTUNA_EPOCH_MAX = 80
OPTUNA_BATCH_CHOICES = [32, 64, 128]
OPTUNA_CUPIKAN = 10
VALIDASI_MIN = 7
PATIENCE_ES = 12
PATIENCE_OPTUNA_SEARCH = 8
PATIENCE_OPTUNA_FINAL = 10

def analisis_pola_angka(data_pasaran, posisi_idx):
    urutan = [baris['angka'][posisi_idx] for baris in data_pasaran]
    n = len(urutan)
    sejak_terakhir = {}
    for d in range(10):
        pos = -1
        for i in range(n-1, -1, -1):
            if urutan[i] == d:
                pos = n - 1 - i
                break
        sejak_terakhir[d] = pos
    awal = max(0, n - JENDELA_PANAS)
    frekuensi = Counter(urutan[awal:])
    status = {}
    for d in range(10):
        hilang = sejak_terakhir[d]
        kali = frekuensi.get(d, 0)
        if hilang < 3 and kali >= AMBANG_KEMBALI:
            status[d] = "panas"
        elif 3 <= hilang <= BAHAYA_TERLALU_LAMA and kali >= 1:
            status[d] = "kembali"
        elif hilang > BAHAYA_TERLALU_LAMA:
            status[d] = "terlambat_bahaya"
        else:
            status[d] = "diam"
    return status, frekuensi, sejak_terakhir

def hitung_bobot_pola(data_pasaran, posisi_idx, prob_murni):
    status, _, _ = analisis_pola_angka(data_pasaran, posisi_idx)
    bobot = np.ones(10, dtype=np.float32)
    for d in range(10):
        s = status[d]
        if s == "panas":
            bobot[d] = 1.30
        elif s == "kembali":
            bobot[d] = 1.15
        elif s == "terlambat_bahaya":
            bobot[d] = 0.60
    prob_baru = prob_murni * bobot
    prob_baru /= np.sum(prob_baru)
    info = {str(d): status[d] for d in range(10)}
    return prob_baru, info

def format_hasil(prob):
    urut = np.argsort(prob)[::-1].tolist()
    return {
        "tujuh": [str(a) for a in urut[:7]],
        "sembilan": [str(a) for a in urut[:9]],
        "tujuh_plus_sisa": [str(a) for a in urut[:7] + [urut[9]]]
    }

def bangun_model(ukuran=LOOKBACK):
    inp = Input(shape=(ukuran, 4))
    x = LSTM(64, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    as_out = Dense(10, activation='softmax', name='as')(x)
    kop_out = Dense(10, activation='softmax', name='kop')(x)
    kep_out = Dense(10, activation='softmax', name='kep')(x)
    eko_out = Dense(10, activation='softmax', name='eko')(x)
    model = Model(inputs=inp, outputs=[as_out, kop_out, kep_out, eko_out])
    model.compile(
        optimizer='adam',
        loss={
            'as': 'sparse_categorical_crossentropy',
            'kop': 'sparse_categorical_crossentropy',
            'kep': 'sparse_categorical_crossentropy',
            'eko': 'sparse_categorical_crossentropy'
        },
        metrics={
            'as': 'accuracy',
            'kop': 'accuracy',
            'kep': 'accuracy',
            'eko': 'accuracy'
        }
    )
    return model

def siapkan_data(dp):
    total = len(dp)
    sampel = total - LOOKBACK
    if sampel < 20 + VALIDASI_MIN:
        return None, None, None, None, 0
    X = np.zeros((sampel, LOOKBACK, 4), dtype=np.float32)
    Y = np.zeros((sampel, 4), dtype=np.int32)
    for i in range(sampel):
        X[i] = [dp[j]['angka'] for j in range(i, i+LOOKBACK)]
        Y[i] = dp[i+LOOKBACK]['angka']
    batas = max(5, int(0.85 * sampel))
    batas = min(batas, sampel - VALIDASI_MIN)
    return (
        X[:batas],
        {'as': Y[:batas,0], 'kop': Y[:batas,1], 'kep': Y[:batas,2], 'eko': Y[:batas,3]},
        X[batas:],
        {'as': Y[batas:,0], 'kop': Y[batas:,1], 'kep': Y[batas:,2], 'eko': Y[batas:,3]},
        batas
    )

def latih_earlystop(X, Y, Xv, Yv):
    if X is None:
        return None, None
    m = bangun_model()
    es = EarlyStopping(patience=PATIENCE_ES, restore_best_weights=True, verbose=0)
    h = m.fit(X, Y, epochs=100, batch_size=32, validation_data=(Xv,Yv), callbacks=[es], verbose=0)
    return m, {'epoch': len(h.history['loss']), 'batch_size': 32, 'berhenti_di': len(h.history['loss'])}

def cari_optuna(X, Y, Xv, Yv):
    if X is None:
        return None, None
    def tujuan(trial):
        m = bangun_model()
        es = EarlyStopping(patience=PATIENCE_OPTUNA_SEARCH, restore_best_weights=True, verbose=0)
        h = m.fit(X, Y, epochs=trial.suggest_int('epoch', OPTUNA_EPOCH_MIN, OPTUNA_EPOCH_MAX),
                  batch_size=trial.suggest_categorical('batch_size', OPTUNA_BATCH_CHOICES),
                  validation_data=(Xv,Yv), callbacks=[es], verbose=0)
        return min(h.history['val_loss'])
    study = optuna.create_study(direction='minimize')
    study.optimize(tujuan, n_trials=OPTUNA_CUPIKAN, show_progress_bar=False)
    bp = study.best_params
    es = EarlyStopping(patience=PATIENCE_OPTUNA_FINAL, restore_best_weights=True, verbose=0)
    m = bangun_model()
    m.fit(X, Y, epochs=bp['epoch'], batch_size=bp['batch_size'],
          validation_data=(Xv,Yv), callbacks=[es], verbose=0)
    return m, {'epoch': bp['epoch'], 'batch_size': bp['batch_size'], 'skor_terbaik': round(study.best_value,8)}

def proses_semua():
    print("Mengambil data...")
    req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        isi = r.read().decode('utf-8')
    mentah = []
    for b in isi.strip().splitlines():
        p = b.split('|')
        if len(p) < 4:
            continue
        an = p[2].strip()
        if len(an) == 4 and an.isdigit():
            mentah.append({
                'pasaran': p[0].strip().upper(),
                'tanggal': p[1].strip(),
                'angka': [int(d) for d in an],
                'nomor': an,
                'waktu': p[3].strip()
            })
    dilihat, bersih = set(), []
    for e in mentah:
        k = (e['pasaran'], e['tanggal'], e['nomor'])
        if k not in dilihat:
            dilihat.add(k)
            bersih.append(e)
    mentah = bersih
    mentah.sort(key=lambda x: (x['tanggal'], x['waktu']))
    per_pasaran = {}
    for e in mentah:
        per_pasaran.setdefault(e['pasaran'], []).append(e)
    for p in per_pasaran:
        if len(per_pasaran[p]) > LIMIT_PER_PASARAN:
            per_pasaran[p] = per_pasaran[p][-LIMIT_PER_PASARAN:]
    daftar = sorted(per_pasaran.keys())
    terbaru = {p: per_pasaran[p][-1] for p in daftar}
    print("\n" + "="*70)
    print("PREDIKSI — TANPA JEBACAN OVERDUE")
    print("="*70)
    for p in daftar:
        print(f" {p:8} | Terakhir: {terbaru[p]['nomor']} | Total: {len(per_pasaran[p])}")
    hasil = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "pengaturan": {
            "LOOKBACK": LOOKBACK,
            "JENDELA_PANAS": JENDELA_PANAS,
            "AMBANG_KEMBALI": AMBANG_KEMBALI,
            "BAHAYA_TERLALU_LAMA": BAHAYA_TERLALU_LAMA,
            "catatan": "Tidak kejar overdue | Prioritas: panas > kembali > diam > dilarang"
        },
        "daftar_pasaran": daftar,
        "hasil": {}
    }
    posisi_nama = ["AS", "KOP", "KEPALA", "EKOR"]
    for p in daftar:
        dp = per_pasaran[p]
        if len(dp) < LOOKBACK + 20 + VALIDASI_MIN:
            print(f"\nDilewati {p} — data kurang")
            continue
        print(f"\nMemproses: {p} | {len(dp)} baris")
        X, Y, Xv, Yv, _ = siapkan_data(dp)
        if X is None:
            continue
        inp_terbaru = np.expand_dims(np.array([dp[j]['angka'] for j in range(-LOOKBACK,0)], dtype=np.float32), 0)
        def jalankan(model, info):
            pred = model.predict(inp_terbaru, verbose=0)
            res = {"pengaturan": info}
            for idx, nm in enumerate(posisi_nama):
                pm = pred[idx][0].copy() / sum(pred[idx][0])
                pp, st = hitung_bobot_pola(dp, idx, pm)
                res[nm] = {
                    "murni": format_hasil(pm),
                    "dipandu_pola": format_hasil(pp),
                    "status": {
                        "panas": [d for d,s in st.items() if s=="panas"],
                        "kembali": [d for d,s in st.items() if s=="kembali"],
                        "diam": [d for d,s in st.items() if s=="diam"],
                        "dilarang": [d for d,s in st.items() if s=="terlambat_bahaya"]
                    }
                }
            return res
        me, ie = latih_earlystop(X, Y, Xv, Yv)
        mo, io = cari_optuna(X, Y, Xv, Yv)
        if not me or not mo:
            continue
        res_e = jalankan(me, ie)
        res_o = jalankan(mo, io)
        hasil["hasil"][p] = {"early_stopping": res_e, "optuna": res_o}
        ekor = res_o["EKOR"]
        print(f"  Optuna: Epoch={io['epoch']} Batch={io['batch_size']}")
        print(f"  EKOR: {''.join(ekor['dipandu_pola']['tujuh_plus_sisa'])}")
        s = ekor['status']
        if s['panas']:
            print(f"  Panas: {s['panas']}")
        if s['kembali']:
            print(f"  Kembali: {s['kembali']}")
        if s['dilarang']:
            print(f"  Dilarang: {s['dilarang']}")
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)
    print("\nSelesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
