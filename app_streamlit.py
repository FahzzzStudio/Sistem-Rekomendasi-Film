import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Sistem Rekomendasi Film",
    page_icon=":poppcorn:"
)

movie_list = joblib.load("movie_list.joblib")
# aslinya butuh file simmilarity.joblib tetapi karean ukurannya 176mb, kita buat on-the fly

cv = CountVectorizer(stop_words="english")
count_matrix = cv.fit_transform(movie_list["tags"].fillna(""))
similarity = cosine_similarity(count_matrix)

titles = movie_list["title"].values

def recommend(movie):
    # cari index dari judul yang dipilih
    index = movie_list[movie_list["title"] == movie].index[0]
    distance = list(enumerate(similarity[index]))
    distance = sorted(distance, reverse=True, key=lambda x: x[1])

    # Aambil film teratas (skip index 0 karna itu filmnya sendiri)
    rekomendasi = [movie_list.iloc[i[0]].title for i in distance[1:6]]

    # Tampilkan sebagai list bernomor
    hasil = "\n".join([f"{idx+1}. {judul}" for idx, judul in enumerate(rekomendasi)])
    st.success(f"5 Rekomendasi Film Teratas:\m\n{hasil}")

st.title(":popcorn: Sistem Rekomendasi Film")
st.markdown("Aplikasi machine learning rekomendasi sistem dengan konsep **Content-based Filtering** dengan **Countvectorizer** dan **Cosine similarity**")

selected_movie = st.selectbox("Film apa yang kamu suka?", titles)

if st.button("Tampilkan Rekomendasi", type="primary"):
    recommend(selected_movie)
    st.balloons()

st.divider()
st.caption("Dibuat dengan :popcorn: oleh **Fahmi Dwi Santoso**")