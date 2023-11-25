import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler
import pickle
from joblib import dump
from sklearn.decomposition import PCA as sklearnPCA
import math

st.title("Analisis dan Prediksi pada Data Audio Emosi")
st.write('Nama : Whinta Virginia Putri')
st.write('NIM : 210411100047')
st.write('Proyek Sains Data A')

data_audio, processing, prediksi  = st.tabs(["Deskripsi Dataset Audio", "Preprocessing dan Modelling", "Prediksi Data Audio"])


with data_audio:
    st.write("## Deskripsi Dataset :")
    st.write("Ada sekitar 200 kata target yang diucapkan dalam frase pembawa 'say the word_' oleh dua aktris (berusia 26 dan 64 tahun) dan rekaman dibuat dari set tersebut menggambarkan tujuh emosi (marah, jijik, takut, bahagia, terkejut menyenangkan, sedih, dan netral). Terdapat total 2800 titik data (file audio).")
    st.write("Kumpulan data ini diatur sedemikian rupa sehingga setiap dari dua aktris perempuan dan emosi mereka terdapat dalam folder tersendiri. Dan di dalamnya, semua file audio 200 kata target dapat ditemukan. Format file audio ini adalah format WAV.")
    st.write("Berikut link kaggle dataset: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download")
    st.write("### Source Aplikasi di Colaboratory :")
    st.write("https://colab.research.google.com/drive/17CkyHgm3dXm4Y9YQwUSFbF67Zhpo4xN1?usp=sharing")


 

with processing:
    st.write("## Pre-processing")
    st.write("Dibuat 11 fitur dari Dataset audio kaggle yaitu Label, Mean, Median, Std_Deviation, Zero_Crossing_Rate, Energy, Spectral_Centroid, Spectral_Bandwidth, Spectral_Rolloff, Chroma, dan MFCCs dari 2191 data, Kemudian di simpan pada format csv.")
    st.write("Berikut adalah hasil Audio_features: ")
    df = pd.read_csv('https://raw.githubusercontent.com/whintaaa/datapsd/main/audio_fitur.csv')
    st.dataframe(df)

    # Memisahkan fitur (X) dan label (y)
    X = df.drop(['Label','Audio_Name'], axis=1) 
    y = df['Label']

    st.write('### Normalisasi Scaler')
    st.write("""
        Normalisasi scaler dalam konteks kode yang Anda berikan tampaknya menggunakan `StandardScaler` dari pustaka scikit-learn untuk melakukan normalisasi pada data. Normalisasi adalah proses mengubah nilai-nilai dalam dataset ke skala umum, biasanya dengan mengurangkan rata-rata dan membagi hasilnya dengan deviasi standar.

        Mari kita bahas langkah-langkah dalam kode tersebut:

        1. **Define and Fit Scaler:**
        ```python
        scaler = StandardScaler()
        scaler.fit(X_train)
        ```
        Pada langkah ini, Anda membuat objek `StandardScaler` dari scikit-learn dan kemudian menggunakan metode `fit` untuk menghitung rata-rata dan deviasi standar dari dataset pelatihan (`X_train`).

        2. **Save Scaler Using Pickle:**
        ```python
        scaler_file_path = r'/content/drive/My Drive/prosaindata/tugas/scaler.pkl'
        with open(scaler_file_path, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        ```
        Pada langkah ini, Anda menyimpan objek scaler ke dalam file menggunakan modul `pickle`. Ini memungkinkan Anda untuk menggunakan kembali objek scaler tanpa harus melatih ulang model setiap kali Anda menjalankan program.

        3. **Transform Training Data Using Scaler:**
        ```python
        X_train_scaled = scaler.transform(X_train)
        ```
        Di sini, Anda menggunakan scaler yang telah dilatih untuk melakukan normalisasi pada dataset pelatihan (`X_train`) dan menyimpan hasilnya dalam `X_train_scaled`.

        4. **Load Scaler Using Pickle:**
        ```python
        with open(r'/content/drive/My Drive/prosaindata/tugas/scaler.pkl', 'rb') as normalisasi:
            loadscal = pickle.load(normalisasi)
        ```
        Langkah terakhir adalah memuat kembali objek scaler dari file yang telah disimpan sebelumnya menggunakan modul `pickle`. Objek scaler yang dimuat disimpan dalam variabel `loadscal`.

        Dengan cara ini, Anda dapat menggunakan objek `loadscal` untuk melakukan transformasi normalisasi pada dataset lain dengan menggunakan parameter normalisasi yang sama seperti yang telah dihitung pada dataset pelatihan.
    """)
   # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    # Define and fit the scaler on the training dataset
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Save the scaler using pickle
    scaler_file_path = r'C:\UTS_PSD\scaler.pkl'
    with open(scaler_file_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    X_train_scaled = scaler.transform(X_train)
    with open(r'C:\UTS_PSD\scaler.pkl', 'rb') as normalisasi:
        loadscal = pickle.load(normalisasi)

    X_test_scaled = loadscal.transform(X_test)

    st.write('X test scaler:', X_test_scaled)

    # Hitung akurasi KNN dari k = 1 hingga 30
    K = 30
    acc = np.zeros((K - 1))

    for n in range(1, K, 2):
        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean").fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc[n - 1] = accuracy_score(y_test, y_pred)

    best_accuracy = acc.max()
    best_k = acc.argmax() + 1

    st.write('## Modelling')
    st.write('### Mencari k terbaik dengan akurasi paling tinggi: ')
    # Tampilkan akurasi terbaik dan nilai k
    st.write('Akurasi terbaik adalah', best_accuracy, 'dengan nilai k =', best_k)

    # Simpan model KNN terbaik
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    best_knn.fit(X_train_scaled, y_train)
    # Save the best KNN model using pickle
    model_file_path = r'C:\UTS_PSD\model.pkl'
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(best_knn, model_file)


    with open(r'C:\UTS_PSD\model.pkl', 'rb') as knn_model:
        load_knn = pickle.load(knn_model)
    y_pred = load_knn.predict(X_test_scaled)
    # Hitung dan tampilkan akurasi KNN
    st.write('Akurasi KNN dengan data test:')
    accuracy = accuracy_score(y_test, y_pred)
    st.write( accuracy)
    # Hitung prediksi label KNN
    knn_predictions = load_knn.predict(X_test_scaled)

    # Simpan hasil prediksi KNN ke dalam DataFrame
    knn_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (KNN)': knn_predictions})

    # Tampilkan tabel prediksi KNN
    st.write("### Tabel Prediksi Label KNN")
    st.dataframe(knn_results_df)

    st.write('### Reduksi PCA')
    st.write("""
        Reduksi PCA (Principal Component Analysis) adalah teknik yang digunakan untuk mengurangi dimensi dari dataset yang kompleks, dengan tetap mempertahankan sebagian besar informasi yang terkandung dalam dataset tersebut. PCA bekerja dengan mentransformasi data asli ke dalam ruang fitur baru yang disebut komponen utama atau principal components. Komponen utama ini diurutkan berdasarkan seberapa besar variansinya, sehingga komponen pertama menyimpan sebagian besar varians, yang diikuti oleh komponen kedua, dan seterusnya.
    """)
    # Lakukan reduksi PCA
    sklearn_pca = sklearnPCA(n_components=10)
    X_train_pca = sklearn_pca.fit_transform(X_train_scaled)
    
    st.write("Principal Components 10:")
    st.write(X_train_pca)

    # Save the PCA model
    pca_model_file_path = r'C:\UTS_PSD\PCA10.pkl'
    with open(pca_model_file_path, 'wb') as pca_model_file:
        pickle.dump(sklearn_pca, pca_model_file)
    # Load the PCA model
    with open(pca_model_file_path, 'rb') as pca_model:
        loadpca = pickle.load(pca_model)

    # Transform test data using the loaded PCA model
    X_test_pca = loadpca.transform(X_test_scaled)
    
    # Continue with KNN and evaluation as needed
    K = 30
    acc_pca = np.zeros((K - 1))
    for n in range(1, K, 2):
        knn_pca = KNeighborsClassifier(n_neighbors=n, metric="euclidean").fit(X_train_pca, y_train)
        y_pred_pca = knn_pca.predict(X_test_pca)
        acc_pca[n - 1] = accuracy_score(y_test, y_pred_pca)

    best_accuracy_pca = acc_pca.max()
    best_k_pca = acc_pca.argmax() + 1

    
    # Tampilkan akurasi terbaik dan nilai k dengan PCA
    st.write('Akurasi KNN terbaik dengan PCA adalah', best_accuracy_pca, 'dengan nilai k =', best_k_pca+1)
    # Hitung prediksi label KNN setelah PCA
    knn_pca_predictions = knn_pca.predict(X_test_pca)

    # Simpan hasil prediksi KNN setelah PCA ke dalam DataFrame
    knn_pca_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (KNN with PCA)': knn_pca_predictions})

    # Tampilkan tabel prediksi KNN setelah PCA
    st.write("### Tabel Prediksi Label KNN dengan PCA")
    st.dataframe(knn_pca_results_df)

with prediksi:
    # Fungsi untuk menghitung fitur audio
    def extract_features(audio_path):
        y, sr = librosa.load(audio_path)

        # Fitur 1: Mean
        mean = np.mean(y)

        # Fitur 2: Median
        median = np.median(y)

        # Fitur 3: Standard Deviation
        std_deviation = np.std(y)

        # Fitur 4: Zero Crossing Rate
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

        # Fitur 5: Energy
        energy = np.mean(librosa.feature.rms(y=y))

        # Fitur 6: Spectral Centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Fitur 7: Spectral Bandwidth
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        # Fitur 8: Spectral Roll-off
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # Fitur 9: Chroma Feature
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

        # Fitur 10: Mel-frequency Cepstral Coefficients (MFCCs)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

        

        return [mean, median, std_deviation, zero_crossing_rate, energy,
                spectral_centroid, spectral_bandwidth, spectral_rolloff, chroma, mfccs]

    def main():
        st.title('Ekstraksi Fitur Audio')
        st.write('Unggah file audio WAV untuk menghitung fitur statistiknya.')

        # Unggah file audio
        uploaded_audio = st.file_uploader("Pilih file audio", type=["wav"])

        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav', start_time=0)

            audio_features = extract_features(uploaded_audio)
            audio_features_reshaped = np.array(audio_features).reshape(1, -1)  # Reshape to 2D array

            feature_names = [
                "Mean", "Median", "Std Deviation", "Zero Crossing Rate", "Energy",
                "Spectral Centroid", "Spectral Bandwidth", "Spectral Rolloff", "Chroma", "MFCCs"
            ]

            # Tampilkan hasil fitur
            st.write("### Hasil Ekstraksi Fitur Audio:")
            for i, feature in enumerate(audio_features):
                st.write(f"{feature_names[i]}: {feature}")

            # Transform audio_features using the loaded scaler
            datauji = loadscal.transform(audio_features_reshaped)
            datapca = loadpca.transform(datauji)
            # Make predictions using the KNN model
            y_pred_uji = load_knn.predict(datauji)
            #y_pred_pca = load_knn.predict(datapca)

            st.write("Fitur-fitur setelah di normalisasi: ", datauji)
            st.write("Data PCA:",datapca)
            #st.write("Predicted Label (PCA):", y_pred_pca)
            st.write("Predicted Label (KNN):", y_pred_uji)
           
            



    if __name__ == "__main__":
        main()


