import openai
import pandas as pd
import json
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity as cosine
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Đọc dữ liệu từ CSV
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)

db = client['cellphoneS']


collections = {
    'phone': db['phone']
}

# Hàm để lấy embedding từ OpenAI (batch API)
def get_embedding(texts, model="text-embedding-ada-002"):
    """
    Gửi các đoạn văn bản đến OpenAI để nhận embedding theo batch.
    """

    try:
        if isinstance(texts, str):
            texts = [texts]  # Chuyển đổi một chuỗi đơn lẻ thành danh sách
        elif not isinstance(texts, list):
            raise ValueError("Input texts phải là danh sách các chuỗi văn bản.")

        # Đảm bảo tất cả phần tử trong 'texts' đều là chuỗi
        texts = [str(text) for text in texts]
        response = openai.embeddings.create(input=texts, model=model)
        return [result.embedding for result in response.data]
    except ValueError as e:
        print(f"Giá trị đầu vào không hợp lệ: {e}")
        print(f"Tài liệu gây lỗi: {texts}")  # In ra tài liệu bị lỗi
        return [None] * len(texts)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Tài liệu gây lỗi: {texts}")  # In ra tài liệu gây lỗi trong trường hợp lỗi khác
        return [None] * len(texts)
    except openai.error.OpenAIError as e:
        print(f"Lỗi khi gọi API OpenAI: {e}")
        return [None] * len(texts)


# Hàm xử lý embedding và lưu vào MongoDB
def process_and_save_embeddings(collection_name, collection):
    """
    Lấy dữ liệu từ collection MongoDB, tạo embedding và lưu lại.
    """
    print(f"Đang xử lý collection '{collection_name}'...")

    # Đọc dữ liệu từ collection MongoDB
    data = list(collection.find())

    # Chuyển đổi sang DataFrame
    df = pd.DataFrame(data)

    if 'content' not in df.columns:
        print(f"Cột 'content' không tồn tại trong collection '{collection_name}', bỏ qua.")
        return

    texts = df['content'].tolist()

    # Xử lý embedding theo batch (chia dữ liệu thành các lô nhỏ)
    batch_size = 50
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Đang xử lý batch {i // batch_size + 1}...")
        batch_embeddings = get_embedding(batch_texts, model="text-embedding-ada-002")
        embeddings.extend(batch_embeddings)

    # Gắn embedding vào DataFrame
    df['embedding'] = embeddings

    # Lưu lại embedding vào MongoDB
    print(f"Đang cập nhật lại MongoDB cho collection '{collection_name}'...")
    for index, row in df.iterrows():
        collection.update_one(
            {'_id': row['_id']},
            {'$set': {'embedding': row['embedding']}}
        )

    print(f"Hoàn thành xử lý collection '{collection_name}'!")

# Thực hiện xử lý cho tất cả các collection
for name, coll in collections.items():
    process_and_save_embeddings(name, coll)

print("Hoàn tất embedding cho tất cả các collections!")
