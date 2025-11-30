# UTS-BigData-Laura
Dashboard klasifikasi dan deteksi objek untuk UTS Pemrograman Big Data.

## 📁 Struktur Folder
```
model/                → berisi file model (disimpan di Google Drive karena >25MB)
sample_images/        → contoh gambar (5 klasifikasi + 5 deteksi)
app.py                → aplikasi Streamlit
requirements.txt      → daftar library
```

## 🔗 Link Model (Google Drive)

Model klasifikasi (.h5):  
👉 [Download di sini](https://drive.google.com/file/d/AAAABBBB/view?usp=sharing)

Model deteksi (.pt):  
👉 [Download di sini](https://drive.google.com/file/d/CCCCDDDD/view?usp=sharing)

> *Model tidak disimpan langsung di GitHub karena ukurannya lebih dari 25MB.*

## 🚀 Menjalankan Streamlit
```
pip install -r requirements.txt
streamlit run app.py
```
