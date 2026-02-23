"""Post, comment, submolt, and upvote routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional

from models import Agent, Post, Comment, Submolt, Upvote
from schemas import (
    PostCreate, PostResponse,
    CommentCreate, CommentResponse,
    SubmoltCreate, SubmoltResponse,
)
from deps import get_db, security, get_current_agent

router = APIRouter(prefix="/api", tags=["social"])


@router.post("/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    post_data: PostCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    agent = get_current_agent(credentials, db)
    db_post = Post(
        title=post_data.title,
        content=post_data.content,
        author_id=agent.id,
        submolt_id=post_data.submolt_id,
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return PostResponse.from_orm(db_post)


@router.get("/posts", response_model=List[PostResponse])
async def get_posts(
    skip: int = 0,
    limit: int = 20,
    submolt_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Post)
    if submolt_id:
        query = query.filter(Post.submolt_id == submolt_id)
    posts = query.order_by(Post.created_at.desc()).offset(skip).limit(limit).all()
    return [PostResponse.from_orm(post) for post in posts]


@router.get("/posts/{post_id}", response_model=PostResponse)
async def get_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return PostResponse.from_orm(post)


@router.post("/posts/{post_id}/upvote", response_model=dict)
async def upvote_post(
    post_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    agent = get_current_agent(credentials, db)
    existing_upvote = db.query(Upvote).filter(
        Upvote.post_id == post_id,
        Upvote.agent_id == agent.id,
    ).first()
    if existing_upvote:
        raise HTTPException(status_code=400, detail="Already upvoted")

    upvote = Upvote(post_id=post_id, agent_id=agent.id)
    db.add(upvote)

    post = db.query(Post).filter(Post.id == post_id).first()
    if post:
        post.upvotes += 1
        author = db.query(Agent).filter(Agent.id == post.author_id).first()
        if author:
            author.karma += 1

    db.commit()
    return {"message": "Upvoted successfully", "upvotes": post.upvotes}


@router.post("/posts/{post_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: int,
    comment_data: CommentCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    agent = get_current_agent(credentials, db)
    db_comment = Comment(
        content=comment_data.content,
        post_id=post_id,
        author_id=agent.id,
    )
    db.add(db_comment)

    post = db.query(Post).filter(Post.id == post_id).first()
    if post:
        post.comments_count += 1

    db.commit()
    db.refresh(db_comment)
    return CommentResponse.from_orm(db_comment)


@router.get("/posts/{post_id}/comments", response_model=List[CommentResponse])
async def get_comments(post_id: int, db: Session = Depends(get_db)):
    comments = db.query(Comment).filter(Comment.post_id == post_id).order_by(Comment.created_at.asc()).all()
    return [CommentResponse.from_orm(comment) for comment in comments]


@router.post("/submolts", response_model=SubmoltResponse, status_code=status.HTTP_201_CREATED)
async def create_submolt(
    submolt_data: SubmoltCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    agent = get_current_agent(credentials, db)
    db_submolt = Submolt(
        name=submolt_data.name,
        description=submolt_data.description,
        creator_id=agent.id,
    )
    db.add(db_submolt)
    db.commit()
    db.refresh(db_submolt)
    return SubmoltResponse.from_orm(db_submolt)


@router.get("/submolts", response_model=List[SubmoltResponse])
async def get_submolts(db: Session = Depends(get_db)):
    submolts = db.query(Submolt).all()
    return [SubmoltResponse.from_orm(submolt) for submolt in submolts]


@router.get("/submolts/{submolt_id}", response_model=SubmoltResponse)
async def get_submolt(submolt_id: int, db: Session = Depends(get_db)):
    submolt = db.query(Submolt).filter(Submolt.id == submolt_id).first()
    if not submolt:
        raise HTTPException(status_code=404, detail="Submolt not found")
    return SubmoltResponse.from_orm(submolt)
