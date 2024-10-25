import psycopg2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request

app = Flask(__name__)

query_likes = 'SELECT l."userId", l."songId", 1 as "like" FROM "Like" as l'
query_play = 'select sp."userId", sp."songId", COUNT(sp."songId") AS listen_count from "SongPlayHistory" as sp left join "Song" as s on s."id" = sp."songId" GROUP BY sp."userId", sp."songId"'

def load_data():
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="pbl6_melodies",
            user="postgres",
            password="290321"
        )
        df_likes = pd.read_sql(query_likes, connection)
        df_plays = pd.read_sql(query_play, connection)

        print("Data loaded successfully")
        return df_likes, df_plays
    except psycopg2.Error as er:
        print("Error: ", er)
    finally:
        if connection:
            connection.close()



# Tạo ma trận người dùng - bài hát từ bảng likes -----------------------------------------------------------------
df_likes, df_plays = load_data()
user_like_matrix = df_likes.pivot(index='userId', columns='songId', values='like').fillna(0)
user_play_matrix = df_plays.pivot(index='userId', columns='songId', values='listen_count').fillna(0)
print("user_play_matrix", user_play_matrix)

# print(user_like_matrix)
user_play_matrix.to_csv('user_play_matrix.csv', index=True)
print("User like matrix has been exported to user_like_matrix.csv")


# Tính ma trận tương đồng người dùng
cosine_sim = cosine_similarity(user_like_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_like_matrix.index, columns=user_like_matrix.index)

cosine_sim_play = cosine_similarity(user_play_matrix)
cosine_sim_play_df = pd.DataFrame(cosine_sim_play, index=user_play_matrix.index, columns=user_play_matrix.index)


# Hàm lấy người dùng tương tự
def get_similar_users(user_id, num_users=5):
    similar_users = cosine_sim_df[user_id].sort_values(ascending=False).drop(user_id).head(num_users)
    return similar_users.index.tolist()

def get_similar_play_users(user_id, num_users=5):
    similar_users = cosine_sim_play_df[user_id].sort_values(ascending=False).drop(user_id).head(num_users)
    return similar_users.index.tolist()


# Hàm đề xuất bài hát
def recommend_songs(user_id, num_recommendations=5):
    similar_users = get_similar_users(user_id)
    print("similar_users", similar_users)
    liked_songs_by_similar_users = user_like_matrix.loc[similar_users].sum(axis=0)
    user_liked_songs = user_like_matrix.loc[user_id]
    
    # Loại bỏ các bài hát đã thích bởi người dùng đầu vào => tránh đề xuất lại bài mà user đã thích
    recommendations = liked_songs_by_similar_users[user_liked_songs == 0].sort_values(ascending=False)
    return recommendations.head(num_recommendations).index.tolist()


def recommend_play_songs(user_id, num_recommendations=5):
    similar_users = get_similar_play_users(user_id)
    played_songs_by_similar_users = user_play_matrix.loc[similar_users].sum(axis=0)
    user_played_songs = user_play_matrix.loc[user_id]
    
    # Loại bỏ các bài hát đã thích bởi người dùng đầu vào => tránh đề xuất lại bài mà user đã thích
    recommendations = played_songs_by_similar_users[user_played_songs == 0].sort_values(ascending=False)
    return recommendations.head(num_recommendations).index.tolist()


# -----------------------------------------------------------------------------------------


# API Flask cho đề xuất bài hát
@app.route('/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    try:
        # Lấy người dùng tương tự
        similar_users = get_similar_users(user_id)
        similar_play_users = get_similar_play_users(user_id)
        
        recommendations = recommend_songs(user_id)
        recommendations_play = recommend_play_songs(user_id)


        return jsonify({
            "similar_users": similar_users,
            "recommended_songs": recommendations,
            "similar_play_users": similar_play_users,
            "recommendations_play": recommendations_play
        })
    except Exception as e:
        return jsonify({"error": str(e)})



# Chạy Flask app
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(port=5555)


# # tạo ma trận người dùng - bài hát từ bảng likes
# user_like_matrix = df_likes.pivot(index='userId', columns='songId', values='like').fillna(0)

# # Tính toán độ tương đồng giữa người dùng
# user_similarity = cosine_similarity(user_like_matrix)
# user_similarity_df = pd.DataFrame(user_similarity, index=user_like_matrix.index, columns=user_like_matrix.index)

# @app.route('/api', methods=['GET'])
# def get_recommendations():
#     user_id = request.args.get('userId')
#     if int(user_id) not in user_similarity_df.index:
#         return jsonify({'error': 'User ID không hợp lệ'})

#     # Lấy danh sách người dùng tương tự
#     similar_users = user_similarity_df[int(user_id)].sort_values(ascending=False).index[1:11]

#     # Lấy danh sách bài hát mà người dùng tương tự đã thích
#     recommended_songs = df_likes[df_likes['userId'].isin(similar_users)]['songId'].unique()

#     return jsonify({'recommended_songs': recommended_songs.tolist()})


# if __name__ == '__main__':
#     app.run(port=5555)