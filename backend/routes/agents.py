"""Agent authentication and profile routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from models import Agent
from schemas import AgentCreate, AgentResponse
from auth import create_access_token
from deps import get_db, security, get_current_agent

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.post("/register", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def register_agent(agent_data: AgentCreate, db: Session = Depends(get_db)):
    existing_agent = db.query(Agent).filter(Agent.username == agent_data.username).first()
    if existing_agent:
        raise HTTPException(status_code=400, detail="Username already exists")

    db_agent = Agent(
        username=agent_data.username,
        display_name=agent_data.display_name,
        bio=agent_data.bio,
        agent_type=agent_data.agent_type,
        bot_type=agent_data.bot_type,
        system_prompt=agent_data.system_prompt,
        model_name=agent_data.model_name or "llama-370b",
    )
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)

    token = create_access_token(data={"sub": db_agent.username, "agent_id": db_agent.id})

    return {
        **AgentResponse.from_orm(db_agent).dict(),
        "access_token": token,
        "token_type": "bearer",
    }


@router.post("/login", response_model=dict)
async def login_agent(agent_data: AgentCreate, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.username == agent_data.username).first()
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": agent.username, "agent_id": agent.id})
    return {
        "access_token": token,
        "token_type": "bearer",
        "agent": AgentResponse.from_orm(agent).dict(),
    }


@router.get("/me", response_model=AgentResponse)
async def get_current_agent_info(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    agent = get_current_agent(credentials, db)
    return AgentResponse.from_orm(agent)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse.from_orm(agent)
