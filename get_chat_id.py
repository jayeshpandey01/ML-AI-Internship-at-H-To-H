#!/usr/bin/env python3
"""
Helper script to get your Telegram Chat ID
"""

import os
import asyncio
from dotenv import load_dotenv
from telegram import Bot

# Load environment variables
load_dotenv('config/config.env')

async def get_chat_id():
    """Get chat ID from recent messages"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not found in config/config.env")
        return
    
    bot = Bot(token=bot_token)
    
    try:
        print("🤖 Getting recent updates...")
        print("📱 Please send a message to your bot first, then run this script")
        
        # Get recent updates
        updates = await bot.get_updates()
        
        if not updates:
            print("❌ No messages found. Please:")
            print("   1. Go to Telegram")
            print("   2. Search for your bot: @pbl_project_bot")
            print("   3. Send any message to the bot")
            print("   4. Run this script again")
            return
        
        print(f"✅ Found {len(updates)} recent messages")
        
        # Get the most recent chat ID
        latest_update = updates[-1]
        chat_id = latest_update.message.chat.id
        
        print(f"🎯 Your Chat ID: {chat_id}")
        print(f"👤 Chat Type: {latest_update.message.chat.type}")
        print(f"📝 Last Message: {latest_update.message.text}")
        
        # Update config file
        print("\n🔧 Updating config/config.env...")
        
        # Read current config
        with open('config/config.env', 'r') as f:
            config_content = f.read()
        
        # Replace the chat ID
        updated_config = config_content.replace(
            'TELEGRAM_CHAT_ID=your_chat_id_here',
            f'TELEGRAM_CHAT_ID={chat_id}'
        )
        
        # Write updated config
        with open('config/config.env', 'w') as f:
            f.write(updated_config)
        
        print("✅ Config updated successfully!")
        print(f"📱 Your bot is ready to send alerts to chat ID: {chat_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📋 Troubleshooting:")
        print("   1. Make sure your bot token is correct")
        print("   2. Send a message to your bot first")
        print("   3. Make sure the bot is not blocked")

def main():
    """Main function"""
    print("📱 TELEGRAM CHAT ID FINDER")
    print("=" * 50)
    
    # Run async function
    asyncio.run(get_chat_id())

if __name__ == "__main__":
    main()