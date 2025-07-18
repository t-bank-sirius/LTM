from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import os
from dotenv import load_dotenv

from .services.memory_service import MemoryService
from .models.schemas import MemoryCreate, MemorySearch, MemoryResponse

load_dotenv()

app = FastAPI(
    title="Long Term Memory API",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_service = MemoryService()

@app.on_event("startup")
async def startup_event():
    await memory_service.initialize()

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Long Term Memory API работает!", "status": "healthy"}

@app.post("/memory/store", response_model=dict, tags=["Memory"])
async def store_memory(memory: MemoryCreate):
    try:
        result = await memory_service.store_memory(
            user_id=memory.user_id,
            content=memory.content,
            context=memory.context
        )
        return {
            "status": "success",
            "message": "Память успешно сохранена",
            "memory_id": result["id"],
            "user_id": memory.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении памяти: {str(e)}")

@app.post("/search", response_model=List[MemoryResponse])
async def search_memories(search_request: MemorySearch):
    try:
        memories = await memory_service.search_memory(
            user_id=search_request.user_id,
            query=search_request.query,
            limit=search_request.limit,
            min_score=search_request.min_score
        )
        return memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске воспоминаний: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8005)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    ) 