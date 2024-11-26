from fastapi import FastAPI, Request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient

# Khởi tạo app
app = FastAPI()

client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/web_project")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_ratings = pd.DataFrame(list(db["UserRating"].find()))
df_anime = pd.DataFrame(list(db["Anime"].find()))

################################## knn

animes_users = df_ratings.pivot(index="anime_id", columns="user_id", values="rating").fillna(0)
mat_anime = csr_matrix(animes_users.values)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model.fit(mat_anime)

def recommender_by_id(anime_id, mat_anime, n):
    # Kiểm tra nếu anime_id có trong dữ liệu
    if anime_id not in df_anime['Anime_id'].values:
        return {"error": "Anime ID không tồn tại"}
    
    idx = df_anime[df_anime['Anime_id'] == anime_id].index[0]
    distance, indices = model.kneighbors(mat_anime[idx], n_neighbors=n)
    recommendations = []
    
    for i in indices.flatten():
        if i != idx:  # Loại bỏ anime hiện tại
            recommendations.append({
                "Anime_id": int(df_anime['Anime_id'].iloc[i]),  # Chuyển đổi sang int
                "Name": str(df_anime['Name'].iloc[i])  # Chuyển đổi sang str
            })
    return recommendations



@app.post("/")
async def recommend(request: Request):
    data = await request.json()
    anime_id = data.get("anime_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    if anime_id is None:
        return {"error": "Vui lòng cung cấp anime_id"}

    # Gọi hàm recommender_by_id
    result = recommender_by_id(anime_id, mat_anime, n)
    return {"recommendations": result} 

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sẽ cung cấp cổng trong biến PORT
    uvicorn.run("knn:app", host="0.0.0.0", port=port)

