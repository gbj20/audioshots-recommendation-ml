import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback

from src.inference.recommend import recommend_for_user

app = FastAPI(title="AudioShots ML Recommendation Service")

class RecommendRequest(BaseModel):
    user_id: str
    limit: int = 10

@app.post("/ml/recommend")
def recommend(req: RecommendRequest):
    try:
        audio_ids = recommend_for_user(req.user_id, req.limit)
        return {
            "recommended_audio_ids": audio_ids
        }
    except ValueError as e:
        # User not found or other validation error
        print(f"ValueError: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Log the full error for debugging
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
def root():
    return {"message": "AudioShots ML Recommendation Service is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}