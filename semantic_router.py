import numpy as np
import openai
from sample_query import productsSample, chitchatSample
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embeddings(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

class Route:
    def __init__(self, name, samples):
        self.name = name
        self.samples = samples

class SemanticRouter:
    def __init__(self, routes, embeddings_file="embeddings.json"):
        self.routes = routes
        self.routes_embedding = {}
        self.embeddings_file = embeddings_file

        if os.path.exists(self.embeddings_file):
            self.load_embeddings()
        else:
            self.generate_embeddings()
            self.save_embeddings()

    def generate_embeddings(self):
        for route in self.routes:
            self.routes_embedding[route.name] = np.array([get_embeddings(sample) for sample in route.samples])

    def save_embeddings(self):
        with open(self.embeddings_file, "w") as f:
            json.dump(self.routes_embedding, f, cls=NumpyEncoder)

    def load_embeddings(self):
        with open(self.embeddings_file, "r") as f:
            self.routes_embedding = json.load(f)
            for key in self.routes_embedding:
                self.routes_embedding[key] = np.array(self.routes_embedding[key])

    def get_routes(self):
        return self.routes

    def guide(self, query):
        query_embedding = np.array(get_embeddings(query))
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores = []

        # Calculate the cosine similarity of the query embedding with the sample embeddings of the router.
        for route in self.routes:
            route_embeddings = self.routes_embedding[route.name]
            route_embeddings = route_embeddings / np.linalg.norm(route_embeddings, axis=1, keepdims=True)
            score = np.mean(np.dot(route_embeddings, query_embedding.T))
            scores.append((score, route.name))

        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[0]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

