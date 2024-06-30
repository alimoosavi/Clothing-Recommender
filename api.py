import settings
from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever

app = FastAPI()
retriever = Retriever(vector_db_uri=settings.VECTOR_DB_URI)


class RecommendationRequestBody(BaseModel):
    description: str


@app.get("/recommend/")
def recommend(body: RecommendationRequestBody):
    desc = body.description
    items = retriever.search_by_text(text_query=desc, top_k=settings.RECOMMENDER_TOP_K)
    return dict(items=items)
