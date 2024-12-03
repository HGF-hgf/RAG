from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from tools import initial_tools
from config import OPENAI_API_KEY
from semantic_router import SemanticRouter, Route
from sample_query import productsSample, chitchatSample

llm = OpenAI(model="gpt-4o-mini", temperature=0)

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=True
)
agent = AgentRunner(agent_worker)

productsRoute = Route("products", productsSample)
chitchatRoute = Route("chitchat", chitchatSample)

# Create a SemanticRouter with the defined routes
semantic_router = SemanticRouter([productsRoute, chitchatRoute])


queries = "So sanh giua iPhone 13 Pro va Samsung Galaxy S21 Ultra"

best_route = semantic_router.guide(queries)
print(f"The best matching route for the query is: {best_route[1]} with a score of {best_route[0]}")

# Sử dụng RouterQueryEngine
response = agent.query(queries)
print(response)