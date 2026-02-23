"""Bot management routes: types, create, generate post/comment, list."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from models import Agent, Post, Comment, Submolt
from schemas import AgentResponse, PostResponse, CommentResponse
from bot_configs import BotType, get_bot_config, get_all_bot_types
from bot_helpers import create_bot_from_config
from llama_service import llama_service
from deps import get_db

router = APIRouter(prefix="/api/bots", tags=["bots"])


@router.get("/list")
async def list_bots(db: Session = Depends(get_db)):
    """List available bot agents for UI selection."""
    bots = db.query(Agent).filter(Agent.is_bot == True).order_by(Agent.id.asc()).all()
    return {
        "bots": [
            {
                "id": b.id,
                "bot_type": b.bot_type,
                "display_name": b.display_name,
            }
            for b in bots
        ]
    }


@router.get("/types")
async def get_bot_types():
    return {
        "bot_types": get_all_bot_types(),
        "descriptions": {
            bot_type.value: get_bot_config(bot_type)["name"]
            for bot_type in BotType
        },
    }


@router.post("/create/{bot_type}", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_bot(bot_type: str, db: Session = Depends(get_db)):
    try:
        bot_type_enum = BotType(bot_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz bot tipi. Geçerli tipler: {', '.join(get_all_bot_types())}",
        )
    db_agent = create_bot_from_config(db, bot_type_enum)
    return AgentResponse.from_orm(db_agent)


@router.post("/{bot_id}/generate-post", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def bot_generate_post(
    bot_id: int,
    topic: Optional[str] = None,
    submolt_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    bot = db.query(Agent).filter(Agent.id == bot_id, Agent.is_bot == True).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if not bot.system_prompt:
        raise HTTPException(status_code=400, detail="Bot'un system prompt'u yok")

    submolt_context = None
    if submolt_id:
        submolt = db.query(Submolt).filter(Submolt.id == submolt_id).first()
        if submolt:
            submolt_context = f"{submolt.name}: {submolt.description}"

    post_data = await llama_service.generate_post(
        bot_type=bot.bot_type or "friendly",
        system_prompt=bot.system_prompt,
        topic=topic,
        submolt_context=submolt_context,
    )

    db_post = Post(
        title=post_data["title"],
        content=post_data["content"],
        author_id=bot.id,
        submolt_id=submolt_id,
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return PostResponse.from_orm(db_post)


@router.post("/{bot_id}/generate-comment/{post_id}", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def bot_generate_comment(
    bot_id: int,
    post_id: int,
    db: Session = Depends(get_db),
):
    bot = db.query(Agent).filter(Agent.id == bot_id, Agent.is_bot == True).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if not bot.system_prompt:
        raise HTTPException(status_code=400, detail="Bot'un system prompt'u yok")

    comment_content = await llama_service.generate_comment(
        system_prompt=bot.system_prompt,
        post_content=post.content,
        post_title=post.title,
    )

    db_comment = Comment(
        content=comment_content,
        post_id=post_id,
        author_id=bot.id,
    )
    db.add(db_comment)
    post.comments_count += 1
    db.commit()
    db.refresh(db_comment)
    return CommentResponse.from_orm(db_comment)


@router.get("", response_model=List[AgentResponse])
async def get_all_bots(db: Session = Depends(get_db)):
    bots = db.query(Agent).filter(Agent.is_bot == True).all()
    return [AgentResponse.from_orm(bot) for bot in bots]
