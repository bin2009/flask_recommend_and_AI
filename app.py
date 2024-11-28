import torch
from flask import Flask, request, jsonify
# from load_model import model, tokenizer
from config import Config
from models import db, User, Comment, Song, Like, SongPlayHistory, Album, AlbumImage, AlbumSong, Artist, Follow
import logging
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from functools import wraps
from http import HTTPStatus
import uuid
from datetime import datetime
import pytz
from sqlalchemy import func, cast, Integer, literal
from flask_cors import CORS
import pandas as pd
import numpy as np
from recommend import DataLoader
from sqlalchemy.orm import joinedload


app = Flask(__name__)
CORS(app) 
app.config.from_object(Config)


# Khởi tạo cơ sở dữ liệu
db.init_app(app)


# khởi chay đề xuất
data_loader = DataLoader()

# with app.app_context():


def get_song(ids):
    # lấy ra các bài hát : bao gồm like, comment, view
    query = db.session.query(Song).options(
        joinedload(Song.album).joinedload(Album.images),
        joinedload(Song.artists).joinedload(Artist.genres),
        joinedload(Song.play_histories)
    )

    # điều kiện query....
    # Điều kiện truy vấn, sắp xếp, và nhóm (nếu có)
    conditions = request.args.get('conditions', {})
    order = request.args.get('order', None)
    group = request.args.get('group', None)
    if conditions:
        query = query.filter_by(**conditions)
    if order:
        query = query.order_by(order)
    if group:
        query = query.group_by(group)

    query = query.filter(Song.id.in_(ids))
    # query = query.limit(10)
    songs = query.all()


    # Chuyển đổi kết quả truy vấn thành danh sách các từ điển để dễ dàng xử lý
    songs_list = []
    for song in songs:
        song_dict = {
            'id': str(song.id),
            'title': song.title,
            'releaseDate': song.releaseDate,
            'duration': song.duration,
            'lyric': song.lyric,
            'filePathAudio': song.filePathAudio,
            'createdAt': song.createdAt,
            # Thêm các thuộc tính bổ sung nếu cần
        }
        if song.album:
            song_dict['album'] = [{
                'albumId': str(alb.albumId),
                'title': alb.title,
                'albumType': alb.albumType,
                'albumImages': [{'image': img.image, 'size': img.size} for img in alb.images]
            } for alb in song.album]
        songs_list.append(song_dict)
        if song.artists:
            song_dict['artists'] = [{
                'id': str(artist.id),
                'name': artist.name,
                # 'main': artist_song.main,
                'genres': [{'genreId': str(genre.genreId), 'name': genre.name} for genre in artist.genres],
                'ArtistSong': {'main': artist_song.main}
            } for artist_song in song.artist_songs for artist in [artist_song.artist]]
        songs_list.append(song_dict)


    return songs_list
    # print("songs: ", songs_list)
    # return jsonify(songs_list)


def get_artist(ids):
    query = db.session.query(Artist).options(
        joinedload(Artist.genres)
    )

    query = query.filter(Artist.id.in_(ids))
    artists = query.all()

    artist_list = []
    for artist in artists:
        artist_dict = {
            'id': str(artist.id),
            'name': artist.name,
            'avatar': artist.avatar,
            'bio': artist.bio,
            'genres': [{'genreId': str(genre.genreId), 'name': genre.name} for genre in artist.genres],
        }
        artist_list.append(artist_dict)
    
    return artist_list

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'status': 'error', 'message': "You're not authenticated"}), HTTPStatus.UNAUTHORIZED
        try:
            token = token.split(' ')[1]
            data = jwt.decode(token, app.config['ACCESS_TOKEN_SECRET'], algorithms=["HS256"])
            print("mã hóa token: ", data)
            current_user = User.query.filter_by(id=data['id']).first()
            if not current_user:
                raise Exception('User not found')
        except ExpiredSignatureError:
            return jsonify({'status': 'error', 'message': 'Token has expired'}), HTTPStatus.FORBIDDEN
        except InvalidTokenError:
            return jsonify({'status': 'error', 'message': 'Invalid token'}), HTTPStatus.FORBIDDEN
        except Exception as e:
            return jsonify({'status': 'error', 'message': 'Access denied', 'error': str(e)}), HTTPStatus.FORBIDDEN
        return f(current_user, *args, **kwargs)
    return decorated


# đề xuất------------------------------------------------


@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([{'id': user.id, 'username': user.username, 'email': user.email} for user in users])


@app.route('/songs', methods=['GET'])
def get_songs():
    songs = Song.query.all()
    return jsonify([
        {
            'id': song.id,
            'title': song.title,
        } for song in songs
    ])

@app.route('/comments', methods=['GET'])
def get_comments():
    comments = Comment.query.all()
    return jsonify([
        {
            'id': comment.id,
            'commentParentId': comment.commentParentId,
            'userId': comment.userId,
            'songId': comment.songId,
            'content': comment.content,
            'hide': comment.hide,
        } for comment in comments
    ])


@app.route('/actions/comment', methods=['POST'])
@token_required
def post_comment(current_user):
    data = request.json  # songId và content
    songId = data.get('songId', '')
    content = data.get('content', '')
    # check songId


    # AI lọc content
    inputs = tokenizer(content, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    output_list = outputs.tolist()
    toxic =  output_list[0][0]

    # Định nghĩa múi giờ Việt Nam
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

    response = {
        'status': '',
        'message': '',
        'score': None
    }

    if float(toxic) > 0.9:
        response['status'] = 'error'
        response['message'] = 'Comment blocked due to community guidelines violation.'
        response['score'] = output_list
        return jsonify(response), HTTPStatus.OK
        # return jsonify({'status': 'error', 'message': 'Comment blocked due to community guidelines violation.', 'score': output_list}), HTTPStatus.OK
    else:
        # tạo mới comment
        new_comment = Comment(
            id=uuid.uuid4(),
            commentParentId=data.get('commentParentId'),
            userId=current_user.id,
            songId=songId,
            content=content,
            hide=False,
            createdAt=datetime.now(vietnam_tz),
            updatedAt=datetime.now(vietnam_tz)
        )
        db.session.add(new_comment)
        db.session.commit()

        # Tạo từ điển cho comment
        comment_data = {
            attr: getattr(new_comment, attr) for attr in ['id', 'commentParentId', 'userId', 'songId', 'content', 'hide', 'createdAt', 'updatedAt']
        }

        response['status'] = 'success'
        response['message'] = 'Comment success'
        response['comment'] = comment_data
        response['score'] = output_list
        return jsonify(response), HTTPStatus.OK
    
        # return jsonify({'status': 'success', 'message': 'Comment success', 'comment': comment_data, 'score': output_list}), HTTPStatus.OK


@app.route('/recommend', methods=['GET'])
@token_required
def recommend(current_user):
    # params: page, page size
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('limit', 10, type=int)


    user_ids = db.session.query(User.id).all()
    likes = db.session.query(Like.userId, Like.songId).all()
    songs = db.session.query(Song.id).all()
    plays = db.session.query(SongPlayHistory.userId, SongPlayHistory.songId).all()
    

    # trọng số 
    like_weight = 1
    play_weight = 0.1


    # Chuyển đổi danh sách user_ids và song_ids thành danh sách
    user_ids = [str(user[0]) for user in user_ids]
    song_ids = [str(song[0]) for song in songs]

    # Khởi tạo ma trận user-song với tất cả giá trị 0
    user_song_matrix = np.zeros((len(user_ids), len(song_ids)), dtype=int)
    user_song_matrix_play = np.zeros((len(user_ids), len(song_ids)), dtype=int)
    user_song_combined_matrix  = np.zeros((len(user_ids), len(song_ids)), dtype=float)
    # print("matrix: ", user_song_matrix)

    # Điền ma trận
    for userId, songId in likes:
        user_index = user_ids.index(str(userId))  # Tìm chỉ số của userId
        song_index = song_ids.index(str(songId))  # Tìm chỉ số của songId
        user_song_combined_matrix[user_index, song_index] += like_weight  # Đặt giá trị 1 nếu user thích bài hát

    for userId, songId in plays:
        user_index = user_ids.index(str(userId))  # Tìm chỉ số của userId
        song_index = song_ids.index(str(songId))  # Tìm chỉ số của songId
        user_song_combined_matrix[user_index, song_index] += play_weight  # Đặt giá trị 1 nếu user thích bài hát

    # Chuyển đổi ma trận thành DataFrame để dễ đọc
    # user_song_df = pd.DataFrame(user_song_matrix, index=user_ids, columns=song_ids)
    # user_song_playdf = pd.DataFrame(user_song_matrix_play, index=user_ids, columns=song_ids)
    user_song_combined_df  = pd.DataFrame(user_song_combined_matrix, index=user_ids, columns=song_ids)


    # user_song_df.to_csv("testhaha.csv")
    user_song_combined_df.to_csv("combined.csv")
    data_loader.set_data(user_matrix=user_song_combined_df)

    songs = data_loader.recommend_songs(user_id=str(current_user.id), page=page, page_size=page_size)
    # songs = data_loader.recommend_songs(user_id=user_id)
    baoloc = get_song(songs)
    return jsonify({"status": 'success', 'message': 'Recommend song success', 'songs': baoloc})


# đề xuất nghệ sĩ dựa trên follow
@app.route('/recommend_artist', methods=['GET'])
@token_required
def recommend_artist(current_user):
    # params: page, page size
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('limit', 10, type=int)

    user_ids = db.session.query(User.id).all()
    artists = db.session.query(Artist.id).all()
    follows = db.session.query(Follow.userId, Follow.artistId).all()

    # tạo ma trận người dùng và nghệ sĩ 
    user_ids = [str(user[0]) for user in user_ids]
    artist_ids = [str(song[0]) for song in artists]
    
    # Tạo ma trận người dùng và nghệ sĩ
    follow_matrix = np.zeros((len(user_ids), len(artist_ids)), dtype=int)
    for userId, artistId in follows:
        user_index = user_ids.index(str(userId))  
        artist_index = artist_ids.index(str(artistId))  
        follow_matrix[user_index, artist_index] = 1 
    follow_df  = pd.DataFrame(follow_matrix, index=user_ids, columns=artist_ids)


    # tính toán đề xuất cosine
    data_loader.set_data_follow(follow_matrix=follow_df)
    artist_ids = data_loader.recommend_artist(user_id=str(current_user.id), page=page, page_size=page_size)

    artists = get_artist(artist_ids)
    print("artist ", artist_ids)
    response = {
        'status': 'success',
        'message': 'Recommend artist success',
        'artists': artists
    }

    return jsonify(response)


    # lấy ra các artist


    



if __name__ == '__main__':
    app.run(debug=True)


