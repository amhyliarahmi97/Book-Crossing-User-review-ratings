
# 📚 Proyek Akhir: Sistem Rekomendasi Buku Menggunakan Book-Crossing Dataset

**Nama:** Rahmi Amilia  
**Platform Dataset:** [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

---

## 1. Project Overview

Sistem rekomendasi merupakan bagian penting dalam membantu pengguna menemukan konten yang relevan di tengah banyaknya pilihan yang tersedia. Di industri buku, pengguna kerap kali kesulitan memilih buku yang sesuai dengan preferensinya. Oleh karena itu, dibutuhkan sistem rekomendasi buku yang mampu memberikan saran secara otomatis berdasarkan minat dan aktivitas pengguna sebelumnya.

Dataset Book-Crossing menyediakan informasi tentang pengguna, buku, dan interaksi pengguna dalam bentuk rating, yang sangat cocok digunakan untuk membangun sistem rekomendasi berbasis Collaborative Filtering.

> 📌 Referensi:  
> - Ziegler, C. et al. “Book-Crossing Dataset.” University of Freiburg, 2004.

---

## 2. Business Understanding

### Problem Statements
1. Pengguna kesulitan menemukan buku yang sesuai dengan minat mereka karena banyaknya pilihan.
2. Tidak semua pengguna memberikan ulasan secara eksplisit, sehingga perlu metode yang mampu mengatasi data yang sparse.

### Goals
- Membangun sistem rekomendasi yang dapat memberikan saran buku kepada pengguna berdasarkan preferensi.
- Menyediakan hasil rekomendasi Top-N untuk masing-masing pengguna.

### Solution Approach
Untuk menyelesaikan masalah ini, dua pendekatan utama digunakan:
- **Collaborative Filtering** menggunakan algoritma SVD dari pustaka `Surprise`.
- **User-Based Filtering** dengan cosine similarity berdasarkan perilaku pengguna.

---

## 3. Data Understanding

Dataset Book-Crossing terdiri dari tiga file utama:

- **BX-Books.csv**: Berisi informasi tentang buku (`ISBN`, `Book-Title`, `Author`, `Year`, dll)
- **BX-Users.csv**: Berisi informasi tentang pengguna (`User-ID`, `Location`, `Age`)
- **BX-Book-Ratings.csv**: Berisi interaksi pengguna dengan buku berupa rating (`User-ID`, `ISBN`, `Book-Rating`)

```python
books.shape
ratings.shape
users.shape
```

Jumlah data:
- Buku: 10.000 (subset)
- Rating: ±1 juta
- Pengguna: ratusan ribu

### Visualisasi & Insight:
- Distribusi rating menunjukkan bahwa sebagian besar rating bernilai 0 (implicit feedback).
- Distribusi usia pengguna cukup beragam, meskipun banyak yang tidak diisi atau outlier.

---

## 4. Data Preparation

Langkah-langkah persiapan data:
1. **Filter rating** untuk hanya mempertahankan rating eksplisit (rating > 0).
2. **Merge** data rating dengan buku dan pengguna.
3. **Normalisasi** dan pembersihan data untuk keperluan modeling.
4. Menghapus data dengan missing value dan outlier pada usia pengguna.

```python
ratings = ratings[ratings['Book-Rating'] > 0]
```

Alasan dilakukan:
- Rating 0 tidak menunjukkan preferensi yang jelas.
- Mengurangi sparsity untuk hasil model yang lebih akurat.

---

## 5. Modeling and Result

### Model 1: Collaborative Filtering dengan SVD

Menggunakan pustaka `Surprise` dan algoritma **SVD** untuk membangun model prediksi rating berdasarkan interaksi pengguna sebelumnya.

```python
from surprise import SVD
model = SVD()
model.fit(trainset)
```

Hasil:  
Model menghasilkan prediksi rating dan mampu memberikan rekomendasi Top-N buku untuk pengguna tertentu.

### Model 2: User-Based Filtering

Menggunakan cosine similarity antar pengguna untuk mengukur kesamaan preferensi, lalu memberikan rekomendasi berdasarkan pengguna yang serupa.

```python
cosine_similarity(user_vector)
```

### Top-N Recommendation

Sistem memberikan 10 buku rekomendasi teratas untuk pengguna berdasarkan model SVD dan User-Based Filtering.

---

## 6. Evaluation

### Metrik Evaluasi: RMSE

Model dievaluasi menggunakan **Root Mean Squared Error (RMSE)** untuk mengukur perbedaan antara rating prediksi dan aktual.

```python
from surprise import accuracy
accuracy.rmse(predictions)
```

Formula RMSE:
\[
	ext{RMSE} = \sqrt{ rac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
\]

Hasil RMSE untuk SVD menunjukkan performa prediksi yang cukup baik dengan nilai di bawah 1.

---

## 7. Conclusion

- Sistem rekomendasi berhasil dibangun menggunakan dua pendekatan berbeda.
- Collaborative Filtering (SVD) lebih efektif untuk data besar dengan banyak interaksi.
- User-Based Filtering dapat digunakan sebagai baseline model.

**Kelebihan:**
- SVD mengatasi masalah sparsity dengan baik.
- Dapat menghasilkan rekomendasi personal yang lebih akurat.

**Kekurangan:**
- User-Based Filtering membutuhkan banyak komputasi saat dataset besar.
- Tidak bisa menangani pengguna baru (cold start problem).

---

## 8. Referensi & Resources

- [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
- Surprise Library: [http://surpriselib.com](http://surpriselib.com)
- [Collaborative Filtering - Towards Data Science](https://towardsdatascience.com/)

