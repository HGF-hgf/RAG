from llama_index.core.tools import FunctionTool
from vector_query_engine import VectorQueryEngine
from embeddings import get_embeddings
from config import phone_collection, laptop_collection, tablet_collection

# Initialize query engines
phone_query_engine = VectorQueryEngine(
    collection=phone_collection,
    embedding_model=get_embeddings,
    vector_index_name="vector_index",
    num_candidates=75,
)

laptop_query_engine = VectorQueryEngine(
    collection=laptop_collection,
    embedding_model=get_embeddings,
    vector_index_name="vector_index",
    num_candidates=68,
)

tablet_query_engine = VectorQueryEngine(
    collection=tablet_collection,
    embedding_model=get_embeddings,
    vector_index_name="vector_index",
    num_candidates=39,
)

# Phone tool
phone_tool = FunctionTool.from_defaults(
    name="query_phone",
    fn=lambda query: phone_query_engine.query(query),
    description="Useful for answering questions about mobile phones, such as iPhone and Android."
)

# Laptop tool
laptop_tool = FunctionTool.from_defaults(
    name="query_laptop",
    fn=lambda query: laptop_query_engine.query(query),
    description="Useful for answering questions about laptops, including brands like Dell and MacBook."
)

# Tablet tool
tablet_tool = FunctionTool.from_defaults(
    name="query_tablet",
    fn=lambda query: tablet_query_engine.query(query),
    description="Useful for answering questions about tablets, such as iPads and Android tablets."
)

initial_tools = [phone_tool, laptop_tool, tablet_tool]