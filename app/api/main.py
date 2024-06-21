# api/main.py
from fastapi import APIRouter
from api.routes import coversongs, models

api_router = APIRouter()

api_router.include_router(coversongs.router, prefix='/coversongs', tags=["coversongs"])
api_router.include_router(models.router, prefix='/models', tags=["models"])