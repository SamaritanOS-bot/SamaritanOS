"""Bot creation and retrieval helpers shared by routes and chain engine."""

from sqlalchemy.orm import Session

from models import Agent
from bot_configs import BotType, get_bot_config


def create_bot_from_config(db: Session, bot_type_enum: BotType) -> Agent:
    config = get_bot_config(bot_type_enum)

    base_username = f"{bot_type_enum.value}_bot"
    username = base_username
    counter = 1
    while db.query(Agent).filter(Agent.username == username).first():
        username = f"{base_username}_{counter}"
        counter += 1

    db_agent = Agent(
        username=username,
        display_name=config["display_name"],
        bio=config["bio"],
        agent_type="LLaMA-370B",
        bot_type=bot_type_enum.value,
        system_prompt=config["system_prompt"],
        model_name="llama-370b",
        is_bot=True,
        verified=True,
    )
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    return db_agent


def get_or_create_bot(db: Session, bot_type_enum: BotType) -> Agent:
    """Get bot; create if missing. Also keep prompt config in sync."""
    bot = db.query(Agent).filter(
        Agent.bot_type == bot_type_enum.value,
        Agent.is_bot == True,
    ).first()

    if bot:
        config = get_bot_config(bot_type_enum)
        changed = False

        if bot.display_name != config["display_name"]:
            bot.display_name = config["display_name"]
            changed = True
        if bot.bio != config["bio"]:
            bot.bio = config["bio"]
            changed = True
        if not (bot.system_prompt or "").strip():
            bot.system_prompt = config["system_prompt"]
            changed = True
        if bot.model_name != "llama3:70b":
            bot.model_name = "llama3:70b"
            changed = True

        if changed:
            db.commit()
            db.refresh(bot)
        return bot

    return create_bot_from_config(db, bot_type_enum)
