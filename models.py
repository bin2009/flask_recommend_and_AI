from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Enum
import uuid
from datetime import datetime

db = SQLAlchemy()

class Comment(db.Model):
    __tablename__ = 'Comment'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    commentParentId = db.Column(UUID(as_uuid=True), nullable=True)
    userId = db.Column(UUID(as_uuid=True), db.ForeignKey('User.id'), nullable=False)
    songId = db.Column(UUID(as_uuid=True), db.ForeignKey('Song.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    hide = db.Column(db.Boolean, default=False, nullable=False)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


    user = db.relationship('User', back_populates='comments')
    song = db.relationship('Song', back_populates='comments')
    reports = db.relationship('Report', back_populates='comment')


class User(db.Model):
    __tablename__ = 'User'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    role = db.Column(Enum('Admin', 'User'), nullable=False)
    username = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False)
    password = db.Column(db.String, nullable=False)
    name = db.Column(db.String, nullable=True)
    image = db.Column(db.String, nullable=True)
    accountType = db.Column(Enum('PREMIUM', 'FREE'), nullable=False, default='Free')
    status = db.Column(Enum('NORMAL', 'LOCK3', 'LOCK7', 'PERMANENT'), nullable=False, default='NORMAL')
    

    comments = db.relationship('Comment', back_populates='user')
    songs = db.relationship('Song', back_populates='uploadUser')
    likes = db.relationship('Like', back_populates='user')
    play_histories = db.relationship('SongPlayHistory', back_populates='user')
    follows = db.relationship('Follow', back_populates='user')
    reports = db.relationship('Report', back_populates='user')



class Song(db.Model):
    __tablename__ = 'Song'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    title = db.Column(db.String, nullable=False)
    duration = db.Column(db.Integer, nullable=True)
    lyric = db.Column(db.String, nullable=True)
    filePathAudio = db.Column(db.String, nullable=False)
    privacy = db.Column(db.Boolean, nullable=False, default=False)
    uploadUserId = db.Column(UUID(as_uuid=True), db.ForeignKey('User.id'), nullable=True)
    releaseDate = db.Column(db.Date, nullable=True)
    image = db.Column(db.String, nullable=True)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    comments = db.relationship('Comment', back_populates='song')
    uploadUser = db.relationship('User', back_populates='songs')
    likes = db.relationship('Like', back_populates='song')
    play_histories = db.relationship('SongPlayHistory', back_populates='song')
    album_songs = db.relationship('AlbumSong', back_populates='song')
    album = db.relationship('Album', secondary='AlbumSong', back_populates='songs')
    artist_songs = db.relationship('ArtistSong', back_populates='song')
    artists = db.relationship('Artist', secondary='ArtistSong', back_populates='songs')

class Like(db.Model): 
    __tablename__ ='Like'

    likeId = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    userId = db.Column(UUID(as_uuid=True), db.ForeignKey('User.id'), nullable=False)
    songId = db.Column(UUID(as_uuid=True), db.ForeignKey('Song.id'), nullable=False)

    user = db.relationship('User', back_populates='likes')
    song = db.relationship('Song', back_populates='likes')

class SongPlayHistory(db.Model):
    __tablename__ = 'SongPlayHistory'

    historyId =  db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    userId = db.Column(UUID(as_uuid=True), db.ForeignKey('User.id'), nullable=False)
    songId = db.Column(UUID(as_uuid=True), db.ForeignKey('Song.id'), nullable=False)
    playtime = db.Column(db.Integer, nullable=True)

    user = db.relationship('User', back_populates='play_histories')
    song = db.relationship('Song', back_populates='play_histories')


class Album(db.Model):
    __tablename__ = 'Album'
    albumId = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    title = db.Column(db.String, nullable=False)
    releaseDate = db.Column(db.Date, nullable=True)
    albumType = db.Column(Enum('album', 'single', 'ep', name='album_type'), nullable=False)

    images = db.relationship('AlbumImage', back_populates='album')
    album_songs = db.relationship('AlbumSong', back_populates='album')
    songs = db.relationship('Song', secondary='AlbumSong', back_populates='album')


class AlbumImage(db.Model):
    __tablename__ = 'AlbumImage'
    
    albumImageId = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    albumId = db.Column(UUID(as_uuid=True), db.ForeignKey('Album.albumId'), nullable=False)
    image = db.Column(db.String, nullable=True)
    size = db.Column(db.Integer, nullable=True)

    album = db.relationship('Album', back_populates='images')


class AlbumSong(db.Model):
    __tablename__ = 'AlbumSong'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    songId = db.Column(UUID(as_uuid=True), db.ForeignKey('Song.id'), nullable=False)
    albumId = db.Column(UUID(as_uuid=True), db.ForeignKey('Album.albumId'), nullable=False)

    song = db.relationship('Song', back_populates='album_songs')
    album = db.relationship('Album', back_populates='album_songs')

class Artist(db.Model):
    __tablename__ = 'Artist'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    name = db.Column(db.String, nullable=False)
    avatar = db.Column(db.Text, nullable=True)
    bio = db.Column(db.Text, nullable=True)
    date = db.Column(db.Date, nullable=True)

    artist_genres = db.relationship('ArtistGenre', back_populates='artist')
    artist_songs = db.relationship('ArtistSong', back_populates='artist')
    songs = db.relationship('Song', secondary='ArtistSong', back_populates='artists')
    genres = db.relationship('Genre', secondary='ArtistGenre', back_populates='artists')
    follows = db.relationship('Follow', back_populates='artist')

class Genre(db.Model):
    __tablename__ = 'Genre'
    
    genreId = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    name = db.Column(db.String, nullable=False)

    artist_genres = db.relationship('ArtistGenre', back_populates='genre')
    artists = db.relationship('Artist', secondary='ArtistGenre', back_populates='genres')


class ArtistGenre(db.Model):
    __tablename__ = 'ArtistGenre'
    
    artistGenreId = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    artistId = db.Column(UUID(as_uuid=True), db.ForeignKey('Artist.id'), nullable=False)
    genreId = db.Column(UUID(as_uuid=True), db.ForeignKey('Genre.genreId'), nullable=False)

    artist = db.relationship('Artist', back_populates='artist_genres')
    genre = db.relationship('Genre', back_populates='artist_genres')


class ArtistSong(db.Model):
    __tablename__ = 'ArtistSong'
    
    artistSongId = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    songId = db.Column(UUID(as_uuid=True), db.ForeignKey('Song.id'), nullable=False)
    artistId = db.Column(UUID(as_uuid=True), db.ForeignKey('Artist.id'), nullable=False)
    main = db.Column(db.Boolean, nullable=False)

    song = db.relationship('Song', back_populates='artist_songs')
    artist = db.relationship('Artist', back_populates='artist_songs')

class Follow(db.Model):
    __tablename__ = 'Follow'
    
    followerId = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    userId = db.Column(UUID(as_uuid=True), db.ForeignKey('User.id'), nullable=False)
    artistId = db.Column(UUID(as_uuid=True), db.ForeignKey('Artist.id'), nullable=False)

    user = db.relationship('User', back_populates='follows')
    artist = db.relationship('Artist', back_populates='follows')


class Report(db.Model):
    __tablename__ = 'Report'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    userId = db.Column(UUID(as_uuid=True), db.ForeignKey('User.id'), nullable=True)
    commentId = db.Column(UUID(as_uuid=True), db.ForeignKey('Comment.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    status = db.Column(Enum('AI', 'PENDING', 'DELETE' , 'NOTDELETE'), nullable=False, default='PENDING')
    createdAt = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = db.relationship('User', back_populates='reports')
    comment = db.relationship('Comment', back_populates='reports')