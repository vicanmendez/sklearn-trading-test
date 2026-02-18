import requests
import json
import sqlite3
import os

# Connect to DB to get a user and a bot
base_dir = os.path.abspath("webapp/instance")
db_path = os.path.join(base_dir, "database.db")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get a bot config
    cursor.execute("SELECT id, name, status FROM bot")
    bots = cursor.fetchall()
    print("Bots found:", bots)
    
    if not bots:
        print("No bots to test delete with.")
        exit()
        
    bot_id = bots[0][0]
    bot_name = bots[0][1]
    
    # Get a user (assuming admin or similar)
    cursor.execute("SELECT id, username FROM user")
    users = cursor.fetchall()
    print("Users found:", users)
    
    conn.close()
    
    print(f"\nAttempting to delete bot {bot_id} ({bot_name})...")
    
    # We need to login first to get a session cookie
    # Assuming there's a login route and we know credentials?
    # Wait, the user is logged in. 
    # Since I don't have the user's password, I can't easily login via script unless I reset it or mock the login.
    # ALTERNATIVE: Use app.test_client() which bypasses network but goes through Flask stack.
    
except Exception as e:
    print(e)
