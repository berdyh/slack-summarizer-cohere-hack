# Setup Guide for Slack RAG Bot

## Prerequisites

1. **Python 3.11+** installed
2. **Cohere API Key** from https://cohere.ai
3. **Slack App** created at https://api.slack.com/apps

## Environment Variables

Create a `.env` file in the project root with:

```bash
# Slack App Configuration
SLACK_CLIENT_ID=your_client_id_here
SLACK_CLIENT_SECRET=your_client_secret_here
SLACK_SIGNING_SECRET=your_signing_secret_here

# Cohere API Key
COHERE_API_KEY=your_cohere_api_key_here

# Base URL for OAuth redirects (use ngrok URL for local development)
BASE_URL=http://localhost:8000
```

## Slack App Configuration

### 1. Create Slack App
1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name: "Slack RAG Bot" (or your choice)
4. Select your workspace

### 2. Configure OAuth & Permissions
Go to **OAuth & Permissions** and add these Bot Token Scopes:
```
channels:history    # Read public channel history
channels:read       # List public channels
groups:history      # Read private channel history (when invited)
files:read          # Download shared files
app_mentions:read   # Detect @mentions
chat:write          # Post responses
commands            # Handle slash commands
```

### 3. Add Slash Command
Go to **Slash Commands** and create:
- **Command**: `/ask`
- **Request URL**: `https://your-ngrok-url.ngrok.io/slack/commands`
- **Description**: "Ask questions about your Slack history"
- **Usage Hint**: "[your question]"

### 4. Configure Event Subscriptions (Optional for MVP)
Go to **Event Subscriptions**:
- **Request URL**: `https://your-ngrok-url.ngrok.io/slack/events`
- **Subscribe to bot events**: (none needed for basic MVP)

### 5. Get Credentials
From **Basic Information**:
- Copy **Client ID** → `SLACK_CLIENT_ID`
- Copy **Client Secret** → `SLACK_CLIENT_SECRET`
- Copy **Signing Secret** → `SLACK_SIGNING_SECRET`

## Installation & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up ngrok (for local development)
```bash
# Install ngrok from https://ngrok.com
ngrok http 8000

# Copy the https URL (e.g., https://abc123.ngrok.io)
# Update your Slack app URLs with this ngrok URL
```

### 3. Run the Application
```bash
python main.py
```

### 4. Test Installation
1. Visit `http://localhost:8000`
2. Click "Add to Slack"
3. Complete OAuth flow
4. Wait for indexing (30-60 seconds)
5. In Slack, try: `/ask What have we been discussing lately?`

## Troubleshooting

### Common Issues:
- **"Invalid signature"**: Check `SLACK_SIGNING_SECRET` is correct
- **"App not installed"**: Complete OAuth flow first
- **"No messages found"**: Wait for indexing, check channel permissions
- **Cohere errors**: Verify `COHERE_API_KEY` is valid

### Debug Mode:
Check `/health` endpoint to see:
- Number of installations
- Total vectors indexed
- Teams that have been indexed

## Next Steps

After basic setup works:
1. Test the `/ask` command in Slack
2. Verify message indexing is working
3. Check that RAG responses are generated
4. Move to Phase 1 implementation (Events API, message processing)
