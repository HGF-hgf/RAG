from pymongo import MongoClient
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")



client2 = MongoClient('mongodb+srv://ngvh1110:1234@cluster0.3f4lo.mongodb.net/')
db2 = client2['newsletter']
collection2 = db2['summerize']

def get_embeddings(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def vector_search(query, collection):
    query_embedding = get_embeddings(query)

    vector_search_stage = {
          "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 182,
            "limit": 4
        }
     }

    project_stage = {
          "$project": {
              "_id": 0,
              "name": 1,
              "price": 1,
              "available_store": 1,
              "product_info": 1,
              "endow": 1,
              "technical_info": 1,
              "score": {"$meta": "vectorSearchScore"}
          }
    }

        # Combine stages into a pipeline
    pipeline = [
        vector_search_stage,
        project_stage
    ]

        # Execute the pipeline
    results = collection.aggregate(pipeline)
    return list(results)


def get_search_results(query):
    get_information = vector_search(query, collection2)

    search_results = ""

    for result in get_information:
        search_results += f"Name: {result['name']}\n"
        search_results += f"Price: {result['price']}\n"
        search_results += f"Technical Info: {result['technical_info']}\n"

    return search_results

query = "so sánh MacBook, và Asus Vivo Book"


def generate_response(query):
    """
    Gọi OpenAI API để tạo phản hồi cho một câu hỏi của khách hàng dựa trên thông tin sản phẩm.

    Args:
        api_key (str): API key của OpenAI.
        query (str): Câu hỏi của khách hàng.
        source_information (str): Thông tin sản phẩm để trả lời câu hỏi.

    Returns:
        str: Phản hồi được tạo ra từ OpenAI API.
    """
    # Đặt API key
    # Prompt được xây dựng
    prompt = (

        f"Câu hỏi : {query}\n"
        f"Thông tin sản phẩm : {search_results}"
        f"Phản hồi : "
    )

    try:
        # Gọi API của OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


    except Exception as e:
        return f"Đã xảy ra lỗi: {e}"

generate_response(query)
