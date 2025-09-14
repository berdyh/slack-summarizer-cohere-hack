"""
Phase 5: Migration Script from In-Memory to Persistent Storage

This script migrates data from the in-memory storage to LanceDB and PostgreSQL.
Run this script after setting up the database to migrate existing data.
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the database manager and main app data
from database import db_manager
from models import SlackMessage
from main import INSTALLATIONS, MESSAGE_VECTORS, USER_CACHE

async def migrate_data():
    """Migrate all in-memory data to persistent storage"""
    print("Starting migration from in-memory to persistent storage...")
    
    try:
        # Initialize database
        await db_manager.initialize()
        print("Database initialized successfully")
        
        # Migrate installations
        print(f"Migrating {len(INSTALLATIONS)} installations...")
        for team_id, installation in INSTALLATIONS.items():
            await db_manager.store_installation(installation)
            print(f"  Migrated installation for team {team_id}")
        
        # Migrate users
        print(f"Migrating {len(USER_CACHE)} users...")
        for (team_id, user_id), user in USER_CACHE.items():
            await db_manager.store_user(user)
            print(f"  Migrated user {user_id} for team {team_id}")
        
        # Migrate message vectors
        total_messages = sum(len(vectors) for vectors in MESSAGE_VECTORS.values())
        print(f"Migrating {total_messages} message vectors...")
        
        for team_id, vectors in MESSAGE_VECTORS.items():
            if vectors:
                # Extract messages and vectors
                messages = []
                vector_arrays = []
                
                for vector_data in vectors:
                    # Convert back to SlackMessage object
                    msg_dict = vector_data["message"]
                    message = SlackMessage(
                        team_id=msg_dict["team_id"],
                        channel_id=msg_dict["channel_id"],
                        user_id=msg_dict["user_id"],
                        text=msg_dict["text"],
                        ts=msg_dict["ts"],
                        thread_ts=msg_dict.get("thread_ts"),
                        message_type=msg_dict.get("message_type", "message"),
                        has_files=msg_dict.get("has_files", False),
                        permalink=msg_dict.get("permalink")
                    )
                    messages.append(message)
                    vector_arrays.append(vector_data["vector"])
                
                await db_manager.store_message_vectors(team_id, messages, vector_arrays)
                print(f"  Migrated {len(messages)} messages for team {team_id}")
        
        print("Migration completed successfully!")
        print(f"Summary:")
        print(f"  - {len(INSTALLATIONS)} installations migrated")
        print(f"  - {len(USER_CACHE)} users migrated")
        print(f"  - {total_messages} message vectors migrated")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

async def verify_migration():
    """Verify that the migration was successful"""
    print("\nVerifying migration...")
    
    try:
        # Check installations
        for team_id in INSTALLATIONS.keys():
            db_installation = await db_manager.get_installation(team_id)
            if db_installation:
                print(f"  ✓ Installation for team {team_id} verified")
            else:
                print(f"  ✗ Installation for team {team_id} not found in database")
        
        # Check users
        for (team_id, user_id) in USER_CACHE.keys():
            db_user = await db_manager.get_user(team_id, user_id)
            if db_user:
                print(f"  ✓ User {user_id} for team {team_id} verified")
            else:
                print(f"  ✗ User {user_id} for team {team_id} not found in database")
        
        # Check message vectors (sample a few)
        for team_id, vectors in MESSAGE_VECTORS.items():
            if vectors:
                # Test search
                test_vector = vectors[0]["vector"]
                results = await db_manager.search_vectors(team_id, test_vector, limit=1)
                if results:
                    print(f"  ✓ Message vectors for team {team_id} verified")
                else:
                    print(f"  ✗ Message vectors for team {team_id} not found in database")
        
        print("Verification completed!")
        
    except Exception as e:
        print(f"Verification failed: {e}")

async def main():
    """Main migration function"""
    print("=== Slack RAG Bot - Database Migration ===")
    print("This script will migrate data from in-memory storage to persistent databases.")
    print("Make sure you have:")
    print("  1. PostgreSQL running and accessible")
    print("  2. LanceDB directory writable")
    print("  3. Environment variables configured")
    print()
    
    # Check if we have data to migrate
    if not INSTALLATIONS and not MESSAGE_VECTORS and not USER_CACHE:
        print("No data found to migrate. Make sure the application has been run and has data.")
        return
    
    print(f"Found data to migrate:")
    print(f"  - {len(INSTALLATIONS)} installations")
    print(f"  - {sum(len(vectors) for vectors in MESSAGE_VECTORS.values())} message vectors")
    print(f"  - {len(USER_CACHE)} users")
    print()
    
    # Confirm migration
    response = input("Do you want to proceed with the migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Run migration
    await migrate_data()
    
    # Verify migration
    await verify_migration()
    
    print("\nMigration process completed!")
    print("You can now restart the application and it will use persistent storage.")

if __name__ == "__main__":
    asyncio.run(main())
