import psycopg2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



class DataLoader:
    def __init__(self):
        self.df_likes = None
        self.df_plays = None
        self.user_like_matrix = None
        self.user_play_matrix = None
        self.cosine_sim_df = None
        self.cosine_sim_play_df = None

        self.follow_matrix = None
        self.cosine_sim_follow_df = None

        self.user_matrix = None

    def set_data(self, user_matrix):
        self.user_matrix = user_matrix
        self.create_matrices()

    def set_data_follow(self, follow_matrix):
        self.follow_matrix = follow_matrix
        self.create_matrices_follow()

    def get_value(self):
        return self.user_like_matrix
    
    def get_value2(self):
        return self.user_play_matrix
    
    
    def create_matrices(self):
        self.cosine_sim_df = pd.DataFrame(cosine_similarity(self.user_matrix), index=self.user_matrix.index, columns=self.user_matrix.index)
        # self.cosine_sim_df.to_csv('cosine_sim_df.csv', index=False)
    
    def create_matrices_follow(self):
        self.cosine_sim_follow_df = pd.DataFrame(cosine_similarity(self.follow_matrix), index=self.follow_matrix.index, columns=self.follow_matrix.index)
        # self.cosine_sim_follow_df.to_csv('cosine_sim_df.csv', index=False)
    
  
    # Hàm lấy người dùng tương tự
    def get_similar_users(self, user_id, num_users=5):
        if user_id not in self.cosine_sim_df.index:
            raise ValueError(f"user_id {user_id} not found in cosine_sim_df")
        
        similar_users = self.cosine_sim_df[user_id].sort_values(ascending=False).drop(user_id).head(num_users)
        return similar_users.index.tolist()
    
    def get_similar_follow_users(self, user_id, num_users=5):
        if user_id not in self.cosine_sim_follow_df.index:
            raise ValueError(f"user_id {user_id} not found in cosine_sim_df")
        
        similar_users = self.cosine_sim_follow_df[user_id].sort_values(ascending=False).drop(user_id).head(num_users)
        return similar_users.index.tolist()


    def recommend_songs(self, user_id, num_recommendations=5, page = 1, page_size = 10):
        print("check user id: ", user_id)
        similar_users = self.get_similar_users(user_id)
        # print("similar_users", similar_users)
        liked_songs_by_similar_users = self.user_matrix.loc[similar_users].sum(axis=0)
        user_liked_songs = self.user_matrix.loc[user_id]
        
        # Loại bỏ các bài hát đã thích bởi người dùng đầu vào => tránh đề xuất lại bài mà user đã thích
        recommendations = liked_songs_by_similar_users[user_liked_songs == 0].sort_values(ascending=False)

        # tính toán phân trang
        start_index = (page-1)*page_size
        end_index = start_index + page_size
        return recommendations.iloc[start_index:end_index].index.tolist()
    
    def recommend_artist(self, user_id, num_recommendations=5, page = 1, page_size = 10):
        # print("check user id: ", user_id)
        similar_users = self.get_similar_follow_users(user_id)
        # print("similar_users", similar_users)
        follow_by_similar_users = self.follow_matrix.loc[similar_users].sum(axis=0)
        user_follow_artist = self.follow_matrix.loc[user_id]
        
        recommendations = follow_by_similar_users[user_follow_artist == 0].sort_values(ascending=False)

        start_index = (page-1)*page_size
        end_index = start_index + page_size
        return recommendations.iloc[start_index:end_index].index.tolist()

