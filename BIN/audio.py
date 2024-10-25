import librosa
import numpy as np
import pandas as pd
import os

# Hàm trích xuất đặc trưng từ file audio
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)

    # Tính toán các đặc trưng âm thanh
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    danceability = min(1.0, tempo / 150.0)
    rmse = librosa.feature.rms(y=y)
    energy = np.mean(rmse)
    loudness = librosa.amplitude_to_db(rmse, ref=np.max).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    speechiness = np.mean(mfcc) / 100.0
    zcr = librosa.feature.zero_crossing_rate(y)
    instrumentalness = 1.0 - np.mean(zcr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = np.argmax(np.mean(chroma, axis=1))
    mode = 1 if np.mean(chroma[key]) > 0.5 else 0
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    liveness = spectral_contrast / 100.0
    duration_ms = librosa.get_duration(y=y, sr=sr) * 1000
    valence = np.mean(mfcc) / 100.0

    return {
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

# Thư mục chứa các file audio
folder_path = "C:/Users/ASUS/Downloads/rac"

# Khởi tạo danh sách để lưu kết quả
features_list = []

# Lặp qua tất cả các file trong thư mục
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Kiểm tra xem file có phải là file audio không (có thể kiểm tra thêm đuôi file nếu cần)
    if os.path.isfile(file_path) and file_name.endswith(('.mp3', '.wav', '.m4a')):
        print(f"Đang xử lý: {file_name}")
        features = extract_audio_features(file_path)
        features['file_name'] = file_name  # Lưu tên file vào kết quả
        features_list.append(features)

# Tạo DataFrame từ danh sách đặc trưng
df = pd.DataFrame(features_list)

# Lưu DataFrame vào file Excel
output_path = "audio_features.xlsx"
df.to_excel(output_path, index=False)
print(f"Lưu đặc trưng vào {output_path} thành công!")
