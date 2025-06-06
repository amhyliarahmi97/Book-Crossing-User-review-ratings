# -*- coding: utf-8 -*-
"""Proyek_Akhir_Book_Crossing_User_review_ratings (2).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Frtl_RKLf7eIA9X_vdvDd7_bCkX8m45w

# Proyek Akhir - Book-Crossing: User Review Ratings

---
By : Rahmi Amilia

### 1. IMPORT LIBRARIES
"""

!pip install numpy==1.24.4

!pip install scikit-surprise

import os
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.accuracy import rmse

import seaborn as sns
import matplotlib.pyplot as plt

"""### 2. LOAD DATA"""

books = pd.read_csv('/content/BX_Books.csv', sep=';', encoding='latin-1', nrows=10000)
ratings = pd.read_csv('/content/BX-Book-Ratings.csv', sep=';', encoding='latin-1')
users = pd.read_csv('/content/BX-Users.csv', sep=';', encoding='latin-1')

print('jumlah data books : ', books.shape)
print('jumlah data ratings : ', ratings.shape)
print('jumlah data users : ', users.shape)

"""### 3. DATA UNDERSTANDING"""

print(books.info())

print(users.info())

print(ratings.info())

"""Statistik Deskriptif"""

print("Statistik Umur Pengguna")
print(users['Age'].describe())

print("Distribusi Rating")
print(ratings['Book-Rating'].value_counts().sort_index())

"""Cek Missing Values"""

print("Missing Values per Dataset:")
print("Books:", books.isnull().sum())
print("\nUsers:", users.isnull().sum())
print("\nRatings:", ratings.isnull().sum())

"""### Data Preprocessing"""

books.rename(columns={
    'ISBN': 'isbn',
    'Book-Title': 'title',
    'Book-Author': 'author',
    'Year-Of-Publication': 'year',
    'Publisher': 'publisher',
    'Image-URL-S': 'img_s',
    'Image-URL-M': 'img_m',
    'Image-URL-L': 'img_l'
}, inplace=True)

print(books.describe)

users.rename(columns={
    'User-ID': 'user_id',
    'Location': 'location',
    'Age': 'age'
}, inplace=True)

print(users.describe)

ratings.rename(columns={
    'User-ID': 'user_id',
    'ISBN': 'isbn',
    'Book-Rating': 'rating'
}, inplace=True)

print(ratings.describe)

print(books.columns)
print(users.columns)
print(ratings.columns)

ratings_books = pd.merge(ratings, books, on='isbn')
print(ratings_books.head())

ratings_books_users = pd.merge(ratings_books, users, on='user_id')
print(ratings_books_users.head())

print(ratings_books_users.info())
print(ratings_books_users.shape)

"""Missing Value"""

books.isnull().sum()

"""### Data Preparation"""

books.isnull().sum()
ratings.isnull().sum()

books = books.dropna()

# Hapus duplikat berdasarkan ISBN
books = books.drop_duplicates(subset='isbn')

# Filter tahun terbit tidak logis
books = books[(books['year'] >= 1900) & (books['year'] <= 2025)]

# Hapus kolom gambar jika ada
books = books.drop(columns=['img_s', 'img_m', 'img_l'], errors='ignore')

display(books.head())
display(books.info())
print(books.isnull().sum())

display(users.head())
display(users.info())
print(users.isnull().sum())

display(ratings.head())
display(ratings.info())

print(ratings.isnull().sum())

"""### 4. EDA"""

plt.figure(figsize=(12, 5))
sns.histplot(data=books, x='year', bins=30, kde=False, color='skyblue')
plt.title('Distribusi Tahun Terbit Buku')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Buku')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 5))
user_rating_counts = ratings['user_id'].value_counts()
sns.histplot(user_rating_counts, bins=50, color='salmon')
plt.title('Distribusi Jumlah Rating per User')
plt.xlabel('Jumlah Rating')
plt.ylabel('Jumlah User')
plt.xlim(0, 100)
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=ratings, x='rating', palette='pastel')
plt.title('Distribusi Nilai Rating')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.show()

"""### 7. MODELING - SVD

### Cosine Similarity
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

books['content'] = books['title'] + ' ' + books['author'] + ' ' + books['publisher']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['content'].fillna(''))

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(books.index, index=books['title']).drop_duplicates()

def content_based_recommendation(title, top_n=5):
    if title not in indices:
        return "Judul tidak ditemukan."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][['title', 'author', 'publisher']]

content_based_recommendation("Classical Mythology")

"""### Training Model SVD"""

reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(ratings[['user_id', 'isbn', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

def get_top_n_recommendations(user_id, books_df, model, n=5):
    user_read_books = ratings[ratings['user_id'] == user_id]['isbn']
    books_not_rated = books_df[~books_df['isbn'].isin(user_read_books)]

    predictions = [model.predict(user_id, isbn) for isbn in books_not_rated['isbn']]

    predictions.sort(key=lambda x: x.est, reverse=True)

    top_books = [pred.iid for pred in predictions[:n]]
    return books_df[books_df['isbn'].isin(top_books)][['title', 'author', 'publisher']]

get_top_n_recommendations(user_id=276725, books_df=books, model=model, n=5)

from collections import defaultdict

def get_top_n(predictions, n=10, threshold=7.0):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        top_n[uid] = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n]
    return top_n

def precision_recall_at_k(predictions, k=10, threshold=7.0):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

precision, recall = precision_recall_at_k(predictions, k=5, threshold=7.0)
print(f'Precision@5: {precision:.4f}, Recall@5: {recall:.4f}')

"""### 5. DATA PREPROCESSING"""

user_ids = ratings.user_id.unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

print(user_ids)
print(user_to_user_encoded)
print(user_encoded_to_user)

book_ids = ratings.isbn.unique().tolist()
book_to_book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded_to_book = {i: x for i, x in enumerate(book_ids)}

print(book_ids)
print(book_to_book_encoded)
print(book_encoded_to_book)

ratings['user'] = ratings.user_id.map(user_to_user_encoded)
ratings['book'] = ratings.isbn.map(book_to_book_encoded)

num_users = len(user_encoded_to_user)
num_books = len(book_encoded_to_book)
print(num_users)
print(num_books)

min_ratings = ratings['rating'].min()
max_ratings = ratings['rating'].max()

print(f'Number of User: {num_users}, Number of Books: {num_books}, Min Rating: {min_ratings}, Max Rating: {max_ratings}')

reader = Reader(rating_scale=(0, 10))

data = Dataset.load_from_df(ratings[['user_id', 'isbn', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)
rmse(predictions)

def get_top_n_recommendations(model, user_id, books_df, ratings_df, n=10):
    buku_yang_belum_dinilai = books_df[~books_df['isbn'].isin(
        ratings_df[ratings_df['user_id'] == user_id]['isbn']
    )]

    buku_yang_belum_dinilai['est_rating'] = buku_yang_belum_dinilai['isbn'].apply(
        lambda x: model.predict(user_id, x).est
    )

    return buku_yang_belum_dinilai.sort_values('est_rating', ascending=False)[['title', 'est_rating']].head(n)

top_books = get_top_n_recommendations(model, user_id=276729, books_df=books, ratings_df=ratings)
print(top_books)

import pickle

with open('model_rekomendasi_svd.pkl', 'wb') as f:
    pickle.dump(model, f)