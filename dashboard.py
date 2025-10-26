import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import os

# --- Load model ---
MODEL_DIR = "model"
DATA_DIR = "data"

vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "feature-bow.p"), "rb"))
model_nb = pickle.load(open(os.path.join(MODEL_DIR, "model-nb.p"), "rb"))

# --- Sidebar menu ---
st.sidebar.title("📊 YouTube Comment NLP Dashboard")
menu = st.sidebar.radio("Pilih Menu:", ["Home", "Dataset & Analysis"])

# ====================================================================
# 🏠 HOME
# ====================================================================
if menu == "Home":
    st.title("🎬 Analisis Sentimen Komentar YouTube")
    st.write("Aplikasi ini menganalisis sentimen dan pola kata dari komentar video YouTube menggunakan NLP dan Machine Learning.")
    
    text_input = st.text_area("Masukkan komentar YouTube di sini:")
    if st.button("Prediksi Sentimen"):
        if text_input.strip() != "":
            X = vectorizer.transform([text_input])
            pred = model_nb.predict(X)[0]
            st.success(f"Hasil Prediksi Sentimen: **{pred}**")
        else:
            st.warning("Masukkan teks terlebih dahulu!")
            
    st.title("ℹ️ Tentang Aplikasi")
    st.write("""
    Aplikasi ini dibuat untuk tugas **Praktikum Aplikasi Web - Crawling dan NLP Dashboard**.  
    Data diambil dari komentar video YouTube menggunakan *YouTube Comment Downloader*.  
    Analisis dilakukan dengan *TF-IDF Vectorizer*, *Naive Bayes Classifier*.
    Video diambil dari Youtuber Bang Windah Batubara
    """)

# ====================================================================
# 📈 DATASET & ANALYSIS
# ====================================================================
elif menu == "Dataset & Analysis":
    st.title("📊 Dataset & Analisis Kata")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "comments_sentiment.csv"))
    except:
        st.error("❌ File 'comments_sentiment.csv' tidak ditemukan. Jalankan dulu tahap prediksi sentimen.")
        st.stop()

    st.subheader("📁 Data Hasil Prediksi Sentimen")
    st.dataframe(df.head())

    # --- Frekuensi Kata ---
    st.subheader("🔠 20 Kata yang Paling Sering Muncul")
    all_words = list(itertools.chain(*[str(text).split() for text in df['clean_comment']]))
    common_words = Counter(all_words).most_common(20)
    freq_df = pd.DataFrame(common_words, columns=['Kata', 'Frekuensi'])
    st.bar_chart(freq_df.set_index('Kata'))

    # --- WordCloud ---
    st.subheader("☁️ WordCloud Komentar")
    text = " ".join(df['clean_comment'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # --- Distribusi Sentimen ---
    st.subheader("📈 Distribusi Sentimen")
    sentiment_counts = df['sentiment'].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
    ax2.set_title("Distribusi Sentimen Komentar YouTube")
    ax2.set_xlabel("Kategori Sentimen")
    ax2.set_ylabel("Jumlah Komentar")
    st.pyplot(fig2)

