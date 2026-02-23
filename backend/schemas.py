from pydantic import BaseModel, ConfigDict
from typing import Optional, Any
from datetime import datetime

# Agent Schemas
class AgentBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    username: str
    display_name: str
    bio: Optional[str] = None
    agent_type: Optional[str] = None
    bot_type: Optional[str] = None
    system_prompt: Optional[str] = None
    model_name: Optional[str] = "llama-370b"

class AgentCreate(AgentBase):
    pass

class AgentResponse(AgentBase):
    id: int
    karma: int
    verified: bool
    is_bot: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Post Schemas
class PostBase(BaseModel):
    title: str
    content: str
    submolt_id: Optional[int] = None

class PostCreate(PostBase):
    pass

class PostResponse(PostBase):
    id: int
    author_id: int
    upvotes: int
    comments_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Comment Schemas
class CommentBase(BaseModel):
    content: str

class CommentCreate(CommentBase):
    pass

class CommentResponse(CommentBase):
    id: int
    post_id: int
    author_id: int
    upvotes: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Submolt Schemas
class SubmoltBase(BaseModel):
    name: str
    description: Optional[str] = None

class SubmoltCreate(SubmoltBase):
    pass

class SubmoltResponse(SubmoltBase):
    id: int
    creator_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Bot Chain Schemas
class BotChainRequest(BaseModel):
    topic: str
    submolt_id: Optional[int] = None
    max_turns: Optional[int] = 6
    seed_prompt: Optional[str] = None
    conversation_id: Optional[str] = None

class BotChainMessage(BaseModel):
    bot_id: int
    bot_type: str
    display_name: str
    content: str

class BotChainMeta(BaseModel):
    intent: str
    risk_level: str
    constraints: list[str]
    route: list[str]
    special_route: Optional[str] = None
    quality_score: Optional[float] = None
    quality_flags: Optional[list[str]] = None

class BotChainResponse(BaseModel):
    topic: str
    order: list[str]
    messages: list[BotChainMessage]
    user_reply: Optional[str] = None
    moltbook_post: Optional[str] = None
    meta: Optional[BotChainMeta] = None


# Content Policy / Action Schemas
class ContentPolicyRequest(BaseModel):
    content: str
    topic_hint: Optional[str] = None
    source_type: Optional[str] = "post"  # post | comment | chat | dm | user_chat | bot_chat | unknown


class ContentPolicyResponse(BaseModel):
    action: str  # approve | reply | ignore | escalate | reject_chat
    reason: str
    confidence: float
    tone: str


class BotActionRequest(BaseModel):
    bot_id: int
    content: str
    topic_hint: Optional[str] = None
    source_type: Optional[str] = "post"  # post | comment | user_chat | bot_chat | dm
    post_id: Optional[int] = None
    conversation_id: Optional[str] = None
    peer_agent_name: Optional[str] = None
    peer_style_hint: Optional[str] = None  # analytical | strategic | narrative | skeptical | neutral
    peer_last_message: Optional[str] = None


class BotActionResponse(BaseModel):
    action: str
    reason: str
    generated_reply: Optional[str] = None


class MemoryAddRequest(BaseModel):
    agent_id: Optional[int] = None
    source_type: str = "post"
    source_id: Optional[str] = None
    intent: str = "observation"
    topic: Optional[str] = None
    claim_text: str
    entities_json: Optional[dict[str, Any]] = None
    outcome_score: Optional[float] = None
    confidence: Optional[float] = None


class MemoryRecord(BaseModel):
    id: int
    agent_id: Optional[int] = None
    source_type: str
    source_id: Optional[str] = None
    intent: str
    topic: Optional[str] = None
    claim_text: str
    entities_json: Optional[dict[str, Any]] = None
    outcome_score: Optional[float] = None
    confidence: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MemoryQueryRequest(BaseModel):
    agent_id: Optional[int] = None
    query: str
    limit: int = 5


class MemorySummaryResponse(BaseModel):
    summary: str
    sample_count: int
    memories: list[MemoryRecord]


class LoreBlockBase(BaseModel):
    key: str
    title: str
    content: str
    is_active: Optional[bool] = True


class LoreBlockUpsertRequest(BaseModel):
    key: str
    title: Optional[str] = None
    content: str
    is_active: Optional[bool] = True


class LoreBlockResponse(LoreBlockBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
