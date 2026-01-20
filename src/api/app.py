# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.append(str(PROJECT_ROOT))

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import traceback

# from src.inference.recommend import recommend_for_user

# app = FastAPI(title="AudioShots ML Recommendation Service")

# class RecommendRequest(BaseModel):
#     user_id: str
#     limit: int = 10

# @app.post("/ml/recommend")
# def recommend(req: RecommendRequest):
#     try:
#         audio_ids = recommend_for_user(req.user_id, req.limit)
#         return {
#             "recommended_audio_ids": audio_ids
#         }
#     except ValueError as e:
#         # User not found or other validation error
#         print(f"ValueError: {str(e)}")
#         raise HTTPException(status_code=404, detail=str(e))
#     except Exception as e:
#         # Log the full error for debugging
#         print(f"Error occurred: {str(e)}")
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/")
# def root():
#     return {"message": "AudioShots ML Recommendation Service is running"}

# @app.get("/health")
# def health():
#     return {"status": "healthy"}


import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import traceback

from src.inference.recommend import recommend_for_user

app = FastAPI(title="AudioShots ML Recommendation Service")

class RecommendRequest(BaseModel):
    user_id: str
    limit: int = 10
    session_id: Optional[str] = None  # NEW: Optional session ID
    enable_randomness: bool = True     # NEW: Control randomness
    exploration_rate: float = 0.2     # NEW: Exploration probability

@app.post("/ml/recommend")
def recommend(req: RecommendRequest):
    try:
        # Generate recommendations with session support
        audio_ids = recommend_for_user(
            user_id=req.user_id,
            top_k=req.limit,
            exploration_rate=req.exploration_rate,
            session_id=req.session_id,
            enable_randomness=req.enable_randomness
        )
        
        return {
            "user_id": req.user_id,
            "recommended_audio_ids": audio_ids,
            "count": len(audio_ids),
            "session_id": req.session_id,
            "exploration_rate": req.exploration_rate
        }
        
    except ValueError as e:
        # User not found or validation error
        print(f"ValueError: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        # Log full error for debugging
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/ml/recommend/hourly")
def recommend_hourly(req: RecommendRequest):
    """
    Recommendations that refresh every hour (Spotify-style)
    Automatically generates session_id based on current hour
    """
    try:
        # Generate hourly session ID
        hourly_session = datetime.now().strftime("%Y%m%d%H")
        
        audio_ids = recommend_for_user(
            user_id=req.user_id,
            top_k=req.limit,
            exploration_rate=req.exploration_rate,
            session_id=hourly_session,
            enable_randomness=True
        )
        
        return {
            "user_id": req.user_id,
            "recommended_audio_ids": audio_ids,
            "count": len(audio_ids),
            "session_id": hourly_session,
            "refresh_strategy": "hourly",
            "next_refresh": datetime.now().replace(minute=0, second=0, microsecond=0).isoformat()
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "AudioShots ML Recommendation Service",
        "version": "2.0",
        "features": [
            "Multi-preference matching",
            "Session-based consistency",
            "Exploration logic",
            "Diversity optimization"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

