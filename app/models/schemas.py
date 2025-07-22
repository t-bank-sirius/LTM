from pydantic import BaseModel, Field
from typing import Optional

class MemoryCreate(BaseModel):
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    context: Optional[str] = Field(None)

class MemorySearch(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(5, ge=1, le=50)
    min_score: float = Field(0.3, ge=0.0, le=1.0)

class MemoryResponse(BaseModel):
    user_id: str
    content: str
    score: float
    time: str
    context: Optional[str] = None
class MemoryStats(BaseModel):
    user_id: str
    total_memories: int
    collection_name: str
    vector_size: int 

class FaceAddRequest(BaseModel):
    user_id: str
    name: str
    image: str

class FaceAddResponse(BaseModel):
    status: str
    message: str
    face_id: str
    user_id: str

class FaceFindRequest(BaseModel):
    user_id: str
    query: str

class FaceFindResponse(BaseModel):
    status: str
    message: str
    image: Optional[str] = None