"""
SMS Span & Fraud Detector (Indonesia)
Author: M. Zidan Maulana
Description: Simple NLP Text Classification using Scikit-learn & Naive Bayes.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def train_spam_model():
    # 1. DATASET LATIH (Training dataset)
    data_sms = [
        # Label 1: Spam / Penipuan
        ("Selamat! Nomor Anda mendapatkan hadiah Rp 50 juta dari Tri. Hubungi wa.me/xxx", 1),
        ("Butuh dana cepat tanpa jaminan? Pinjaman online bunga 0% klik link bit.ly/xxx", 1),
        ("PROMO DEPOSIT slot gacor maxwin hari ini, bonus new member 100%", 1),
        ("Info resmi PLN: ID pelanggan Anda menang undian Rp 10jt, klaim di xxx", 1),
        ("Maaf Mengganggu Waktunya Pak/Ibu Kami Dari KOPERASI Menawarkan PINJAMAN-ONLINE 5jt Sampai 500jt Poses Cepat Bunga 2% Pertahun Info Whatsapp :0823-1757-2717", 1),
        ("Hanya Kk & Ktp Sudah Bisa buat modal usaha dengan bunga% Min 5-500jt Melayani Seluruh Indonesia Minat WA:087844302111 tks...", 1),
        ("Info Pinjaman Tunai Cepat Cair Dan Terpercaya Tampa Agunan Tampa Riba Info Chat WA:085248724234", 1),
        # Label 0: SMS Normal / Ham
        ("Zidan, jangan lupa kerjakan tugas RPL dan persiapkan projek minggu depan ya", 0),
        ("Paket Anda sedang dibawa oleh kurir menuju ke alamat tujuan", 0),
        ("Nanti sore nongkrong di warung kopi depan sekolah tidak?", 0),
        ("Kuota internet Anda sisa 1GB. Segera lakukan isi ulang.", 0),
        ("Zidan, kemarin ada tugas tidak dari pak Wali?", 0)
    ]

    df = pd.DataFrame(data_sms, columns=["teks_sms", "label"])
    X_train = df["teks_sms"]
    Y_train = df["label"]

    # 2. MODEL PIPELINE (TF-IDF Vectorizer + Naive-Bayes Clasifier)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # 3. TRAINING MODEL
    print("[INFO] Melatih model AI...")
    model.fit(X_train, Y_train)
    print("[INFO] Pelatihan selesai!")

    return model

def predict_sms(model, text_list):
    result = model.predict(text_list)
    for text, pred in zip(text_list, result):
        status = "PENIPUAN / SPAM" if pred == 1 else "SMS normal"
        print(f"Pesan: \"{text}\"")
        print(f"Status: [{status}]\n")

if __name__ == "__main__":
    ai_model = train_spam_model()

    play = True

    while play:
        tes = input("Apakah ingin melakukan uji coba? ")
        if tes == "tidak" or tes == "tutup":
            print("Terimakasih sudah bermain!")
            play = False
            break
        else:
            data1 = input("Masukkan teks uji coba 1 ")
            data2 = input("Masukkan teks uji coba 2 ")

        # data uji coba yang belum dilihat model
        data_uji = [
            data1,
            data2
        ]

        print("==== HASIL UJI COBA AI ====")
        predict_sms(ai_model, data_uji)