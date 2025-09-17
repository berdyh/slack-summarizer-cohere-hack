"""
Phase 5: Database Setup Script

This script sets up the PostgreSQL database and LanceDB for the Slack RAG system.
Run this script before running the main application for the first time.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def setup_database():
    """Set up the database tables and LanceDB"""
    print("Setting up database for Slack RAG Bot...")
    
    try:
        # Import database manager
        from database import db_manager
        
        # Initialize database (this will create tables)
        await db_manager.initialize()
        
        print("Database setup completed successfully!")
        print("You can now run the main application.")
        
    except Exception as e:
        print(f"Database setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your DATABASE_URL in .env file")
        print("3. Ensure the database user has CREATE privileges")
        print("4. Make sure the LanceDB directory is writable")
        import traceback
        traceback.print_exc()

async def check_environment():
    """Check if environment variables are properly configured"""
    print("Checking environment configuration...")
    
    required_vars = [
        "SLACK_CLIENT_ID",
        "SLACK_CLIENT_SECRET", 
        "SLACK_SIGNING_SECRET",
        "COHERE_API_KEY"
    ]
    
    optional_vars = [
        "DATABASE_URL",
        "LANCE_DB_PATH"
    ]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        print(f"Missing required environment variables: {', '.join(missing_required)}")
        print("Please add these to your .env file")
        return False

    print("All required environment variables are set")

    # Check optional variables
    for var in optional_vars:
        if not os.getenv(var):
            print(f"Optional variable {var} not set, using default")
    
    return True

async def main():
    """Main setup function"""
    print("=== Slack RAG Bot - Database Setup ===")
    print()
    
    # Check environment
    if not await check_environment():
        return
    
    print()
    
    # Set up database
    await setup_database()

if __name__ == "__main__":
    asyncio.run(main())
