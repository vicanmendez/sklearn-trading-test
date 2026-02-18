import sys
import os
sys.path.append(os.path.abspath('webapp'))
from app import app, db
from db_models import Bot
import traceback

with app.app_context():
    print("App context pushed.")
    try:
        # Get bot with ID 4 (one of the no_model ones)
        bot = Bot.query.get(4)
        if not bot:
            print("Bot 4 not found.")
            # Try finding any bot
            bot = Bot.query.first()
            if not bot:
                 print("No bots found at all.")
                 exit()
        
        print(f"Attempting to delete bot {bot.id} ({bot.name})...")
        
        db.session.delete(bot)
        db.session.commit()
        print("Delete successful!")
        
    except Exception:
        traceback.print_exc()
