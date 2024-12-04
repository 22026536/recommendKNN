from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from bson import ObjectId

# Khởi tạo app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối MongoDB
client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_favorites = pd.DataFrame(list(db["UserFavorites"].find()))  # Thay đổi thành UserFavorites
df_anime = pd.DataFrame(list(db["Anime"].find()))

# Chuyển đổi ObjectId trong DataFrame
# df_favorites['_id'] = df_favorites['_id'].astype(str)
# df_favorites['User_id'] = df_favorites['User_id'].astype(str)
df_favorites['favorites'] = df_favorites['favorites'].apply(lambda x: [str(i) for i in x])  # Chuyển đổi các Anime_id trong favorites thành string

# df_anime['_id'] = df_anime['_id'].astype(str)
# df_anime['Anime_id'] = df_anime['Anime_id'].astype(str)

################################## KNN

# Chuyển đổi dữ liệu từ UserFavorites thành ma trận người dùng - anime
user_anime_matrix = pd.DataFrame(
    [
        (user["User_id"], anime_id, 1)
        for user in df_favorites.to_dict(orient="records")
        for anime_id in user["favorites"]
    ],
    columns=["User_id", "Anime_id", "Rating"]
)

# Pivot bảng để tạo ma trận sparse
animes_users = user_anime_matrix.pivot(index="Anime_id", columns="User_id", values="Rating").fillna(0)
mat_anime = csr_matrix(animes_users.values)

# Huấn luyện mô hình KNN
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model.fit(mat_anime)

# Hàm gợi ý anime theo anime_id
def recommender_by_id(anime_id, mat_anime, n):
    if anime_id not in animes_users.index:
        return {"error": "Anime ID không tồn tại"}

    idx = animes_users.index.get_loc(anime_id)
    distances, indices = model.kneighbors(mat_anime[idx], n_neighbors=n)
    recommendations = []

    for i in indices.flatten():
        if i != idx:  # Loại bỏ anime hiện tại
            anime_data = df_anime[df_anime['Anime_id'] == animes_users.index[i]].iloc[0].to_dict()
            recommendations.append(anime_data)
    return recommendations

# Hàm chuyển đổi ObjectId thành dạng JSON serializable
def jsonable(data):
    if isinstance(data, list):
        return [jsonable(item) for item in data]
    elif isinstance(data, dict):
        return {key: jsonable(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data

# API POST để gợi ý anime
@app.post("/")
async def recommend(request: Request):
    data = await request.json()
    anime_id = data.get("anime_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    if anime_id is None:
        return {"error": "Vui lòng cung cấp anime_id"}

    # Gọi hàm recommender_by_id
    result = jsonable(recommender_by_id(anime_id, mat_anime, n))
    return result

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sẽ cung cấp cổng trong biến PORT
    uvicorn.run("knn:app", host="0.0.0.0", port=port)
