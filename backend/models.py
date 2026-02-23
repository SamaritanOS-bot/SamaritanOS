from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    display_name = Column(String(150), nullable=False)
    bio = Column(Text, nullable=True)
    agent_type = Column(String(64), nullable=True)  # GPT-4, Claude, etc.
    bot_type = Column(String(64), nullable=True)  # creative, analytical, humorous, etc.
    system_prompt = Column(Text, nullable=True)  # Bot's system prompt
    model_name = Column(String(100), default="llama-370b")  # Model used
    karma = Column(Integer, default=0)
    verified = Column(Boolean, default=False)
    is_bot = Column(Boolean, default=False)  # Is this an automated bot?
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    posts = relationship("Post", back_populates="author")
    comments = relationship("Comment", back_populates="author")
    created_submolts = relationship("Submolt", back_populates="creator")
    memories = relationship("AgentMemory", back_populates="agent", cascade="all, delete-orphan")
    memory_summaries = relationship("MemorySummary", back_populates="agent", cascade="all, delete-orphan")

class Submolt(Base):
    __tablename__ = "submolts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    creator_id = Column(Integer, ForeignKey("agents.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    creator = relationship("Agent", back_populates="created_submolts")
    posts = relationship("Post", back_populates="submolt")

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey("agents.id"))
    submolt_id = Column(Integer, ForeignKey("submolts.id"), nullable=True)
    upvotes = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    author = relationship("Agent", back_populates="posts")
    submolt = relationship("Submolt", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    upvote_list = relationship("Upvote", back_populates="post", cascade="all, delete-orphan")

class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    post_id = Column(Integer, ForeignKey("posts.id"))
    author_id = Column(Integer, ForeignKey("agents.id"))
    upvotes = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    post = relationship("Post", back_populates="comments")
    author = relationship("Agent", back_populates="comments")

class Upvote(Base):
    __tablename__ = "upvotes"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"))
    agent_id = Column(Integer, ForeignKey("agents.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    post = relationship("Post", back_populates="upvote_list")
    agent = relationship("Agent")


class AgentMemory(Base):
    __tablename__ = "agent_memories"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True, index=True)
    source_type = Column(String(32), nullable=False, index=True)  # post | comment | dm | chain
    source_id = Column(String(64), nullable=True)
    intent = Column(String(32), nullable=False, index=True)  # question | objection | agreement | challenge
    topic = Column(String(255), nullable=True)
    claim_text = Column(Text, nullable=False)
    entities_json = Column(JSON, nullable=True)
    outcome_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    agent = relationship("Agent", back_populates="memories")


class MemorySummary(Base):
    __tablename__ = "memory_summaries"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True, index=True)
    summary_scope = Column(String(32), nullable=False, default="rolling")
    summary_text = Column(Text, nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    agent = relationship("Agent", back_populates="memory_summaries")


class LoreBlock(Base):
    __tablename__ = "lore_blocks"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(64), unique=True, index=True, nullable=False)  # canon_core | canon_world | canon_style
    title = Column(String(160), nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
