import librosa
import numpy as np

# Load audio file
y, sr = librosa.load("C:/Users/ASUS/Downloads/ChayNgayDiSkyTour2019.wav")


def calculate_acousticness(audio_path):
    # Tải audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Trích xuất các đặc trưng
    # Sử dụng một phương pháp đơn giản để ước lượng acousticness
    # Bạn có thể sử dụng các thuật toán phức tạp hơn nếu cần
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    # Tính toán acousticness dựa trên các đặc trưng đã trích xuất
    # Đây là một phương pháp đơn giản, bạn có thể điều chỉnh công thức theo nhu cầu
    acousticness = (spectral_centroid / (spectral_centroid + spectral_bandwidth))

    # Đảm bảo giá trị nằm trong khoảng [0, 1]
    acousticness = np.clip(acousticness, 0, 1)
    
    return acousticness

# 1. Loudness: độ lớn trung bình của bài hát


# 2. acousticness: độ âm thanh

# 2. Danceability: Dựa trên nhịp điệu và độ ổn định của nhịp
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
danceability = min(1.0, tempo / 150.0)  # Ước lượng danceability

# 3. Duration (ms)
duration_ms = librosa.get_duration(y=y, sr=sr) * 1000

# 4. Energy: Dựa trên Root Mean Square Energy
rmse = librosa.feature.rms(y=y)
energy = np.mean(rmse)

# 5. Instrumentalness: Gần đúng qua đánh giá zero-crossing rate (tính ổn định tín hiệu)
zcr = librosa.feature.zero_crossing_rate(y)
instrumentalness = 1.0 - np.mean(zcr)  # Ước tính độ "instrumental"

# 6. Key và Mode (Lấy key chính từ phổ và độ lệch)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
key = np.argmax(np.mean(chroma, axis=1))
mode = 1 if np.mean(chroma[key]) > 0.5 else 0  # Ước lượng mode (Major/Minor)

# 7. Liveness: Có thể đo lường qua spectral contrast
spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
liveness = spectral_contrast / 100.0  # Liveness tỷ lệ hóa

# 8. Loudness: Tính dB từ RMSE
loudness = librosa.amplitude_to_db(rmse, ref=np.max).mean()

# 9. Speechiness: Tỷ lệ phần trăm phổ âm trung, phù hợp với giọng nói
speechiness = np.mean(librosa.feature.mfcc(y=y, sr=sr)) / 100.0

# 10. Tempo: Đã tính ở trên
tempo = tempo

# 11. Valence: Đánh giá cảm xúc dựa trên phổ âm và âm sắc
valence = np.mean(librosa.feature.mfcc(y=y, sr=sr)) / 100.0

# Đặt các kết quả vào dict
audio_features = {
    "acousticness": None,  # Giá trị cần ML hoặc thuật toán đặc thù
    "danceability": danceability,
    "duration_ms": duration_ms,
    "energy": energy,
    "instrumentalness": instrumentalness,
    "key": key,
    "liveness": liveness,
    "loudness": loudness,
    "mode": mode,
    "speechiness": speechiness,
    "tempo": tempo,
    "valence": valence,
}

print(audio_features)
