"""
SMS Span & Fraud Detector (Indonesia)
Author: M. Zidan Maulana
Description: Simple NLP Text Classification using Scikit-learn & Naive Bayes.
"""

import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Inisialisasi Stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text: str) -> str:
    """Membersihkan teks: lowercase, hapus angka, hapus simbol, dan stemming."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = stemmer.stem(text)
    return text


def load_and_prepare_data(filepath: str):
    """Membaca file CSV dan melakukan preprocessing pada seluruh dataset."""
    print(f"[INFO] Membaca dataset dari {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] File {filepath} tidak ditemukan!")
        return None, None

    print("[INFO] Membersihkan dan memproses teks dengan Sastrawi...")
    df["clean_teks"] = df["teks_sms"].apply(preprocess_text)

    X = df["clean_teks"]
    y = df["label"]
    return X, y

def train_model(X, y):
    """Melatih model AI dan mengevaluasi akurasinya menggunakan Train-Test Split."""
    # Split data 80% Latih, 20% Uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: TF-IDF (1-2 N-grams) + Naive Bayes Classifier
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        MultinomialNB(),
    )

    print("[INFO] Melatih model AI...")
    model.fit(X_train, y_train)

    # Evaluasi Performa
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[EVALUASI] Akurasi Model: {acc * 100:.2f}%")
    print("Detail Laporan Klasifikasi:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Normal", "Spam/Penipuan"]
        )
    )

    return model


def predict_single_text(model, raw_text: str) -> str:
    """Memproses satu kalimat input dan mengembalikan status prediksi."""
    cleaned_text = preprocess_text(raw_text)
    pred = model.predict([cleaned_text])[0]
    return "PENIPUAN / SPAM" if pred == 1 else "SMS NORMAL"


def interactive_cli(model):
    """Menu interaktif berbasis terminal untuk menguji input pengguna."""
    print("=" * 50)
    print("      DETEKTOR SMS PENIPUAN / SPAM (INDONESIA)      ")
    print("=" * 50)
    print("Ketik pesan SMS yang ingin kamu tes.")
    print("Ketik 'keluar' atau 'exit' untuk menghentikan program.\n")

    while True:
        try:
            user_input = input("Masukkan Teks SMS > ").strip()

            # Kondisi keluar dari loop
            if user_input.lower() in ["keluar", "exit"]:
                print("\n[INFO] Terima kasih telah menggunakan program ini. Sampai jumpa!")
                break

            # Validasi jika input kosong
            if not user_input:
                print("[!] Teks tidak boleh kosong. Silakan coba lagi.\n")
                continue

            # Prediksi dan tampilkan hasil
            status = predict_single_text(model, user_input)
            print(f"-> Hasil Analisis AI: [{status}]\n")

        except KeyboardInterrupt:
            print("\n\n[INFO] Program dihentikan.")
            break


def main():
    # 1. Load Data
    X, y = load_and_prepare_data("dataset.csv")
    if X is None or y is None:
        return

    # 2. Train Model
    ai_model = train_model(X, y)

    # 3. Jalankan CLI Interaktif
    interactive_cli(ai_model)


if __name__ == "__main__":
    main()