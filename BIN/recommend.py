import psycopg2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request

app = Flask(__name__)

# kết nối db
try:
    connection = psycopg2.connect(
        host="localhost",
        database="pbl6_melodies",
        user="postgres",
        password="290321"
    )
    query = 'select * from "Song"'
    df_song = pd.read_sql(query, connection)
    print("Test", df_song.head())
except psycopg2.error as er:
    print("Error: ", er)
finally:
    if connection:
        connection.close()

# lấy feature từ song
features = ['title', 'duration', 'releaseDate']

def combinedFeatures(row):
    return str(row['title']) + " " + str(row['duration']) + " " + str(row['releaseDate'])

df_song['combinedFeatures'] = df_song.apply(combinedFeatures, axis = 1)

print(df_song['combinedFeatures'].head())

# tfidf
tf = TfidfVectorizer()
tfMatrix = tf.fit_transform(df_song['combinedFeatures'])

# tính độ tương đồng
similar = cosine_similarity(tfMatrix)

number = 10
@app.route('/api', methods=['GET'])
def get_data():
    ket_qua = []
    songId = request.args.get('id')
    # songId = int(songId)

    if songId not in df_song['id'].values:
        return jsonify({'Lỗi': 'Id k hợp lệ'})
    
    indexSong = df_song[df_song['id'] == songId].index[0]
    similarSong = list(enumerate(similar[indexSong]))

    sortedSimilarSong = sorted(similarSong, key=lambda x: x[1], reverse=True)

    def get_song(index):
        return df_song[df_song.index == index]['title'].values[0]

    for i in range(1, number + 1):
        ket_qua.append(get_song(sortedSimilarSong[i][0]))

    print(ket_qua)

    data = {'nhạc gợi ý là: ': ket_qua}
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=5555)