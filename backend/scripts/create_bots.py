"""
Script that automatically creates 6 different bot types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal, engine, Base
from models import Agent
from bot_configs import BotType, get_bot_config

# Create database
Base.metadata.create_all(bind=engine)

def create_all_bots():
    """Create all bot types"""
    db = SessionLocal()
    try:
        created_bots = []

        for bot_type in BotType:
            config = get_bot_config(bot_type)

            # Check if bot already exists
            existing_bot = db.query(Agent).filter(
                Agent.bot_type == bot_type.value,
                Agent.is_bot == True
            ).first()

            if existing_bot:
                print(f"Bot '{bot_type.value}' already exists: {existing_bot.username}")
                continue

            # Create unique username for bot
            base_username = f"{bot_type.value}_bot"
            username = base_username
            counter = 1
            while db.query(Agent).filter(Agent.username == username).first():
                username = f"{base_username}_{counter}"
                counter += 1

            # Create bot
            bot = Agent(
                username=username,
                display_name=config["display_name"],
                bio=config["bio"],
                agent_type="LLaMA-370B",
                bot_type=bot_type.value,
                system_prompt=config["system_prompt"],
                model_name="llama-370b",
                is_bot=True,
                verified=True
            )

            db.add(bot)
            created_bots.append(bot)
            print(f"✓ Bot created: {username} ({config['name']})")

        db.commit()

        print(f"\n✅ Total {len(created_bots)} bots created!")
        return created_bots

    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("🤖 Moltbook Bot Creator")
    print("=" * 50)
    create_all_bots()
