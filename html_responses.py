"""
HTML Response Templates for Slack RAG Bot
Contains all HTML templates and response functions to keep main.py clean
"""

from fastapi.responses import HTMLResponse
from typing import Dict, Any

def home_page() -> HTMLResponse:
    """Landing page with 'Add to Slack' button"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Slack RAG Bot - AI-Powered Knowledge Assistant</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 40px 20px; 
                line-height: 1.6;
                color: #333;
            }
            .header { text-align: center; margin-bottom: 40px; }
            .btn { 
                background: #4A154B; 
                color: white; 
                padding: 15px 30px; 
                text-decoration: none; 
                border-radius: 6px; 
                display: inline-block;
                font-weight: 600;
                transition: background 0.2s;
            }
            .btn:hover { background: #350d36; }
            .feature { 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 8px; 
                border-left: 4px solid #4A154B;
            }
            .status { 
                background: #e8f5e8; 
                padding: 15px; 
                margin: 20px 0; 
                border-radius: 6px; 
                border: 1px solid #4caf50;
            }
            code { 
                background: #f1f3f4; 
                padding: 2px 6px; 
                border-radius: 3px; 
                font-family: 'Monaco', 'Menlo', monospace;
            }
            .demo-commands {
                background: #fff3cd;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #ffeaa7;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Slack RAG Bot</h1>
            <p style="font-size: 1.2em; color: #666;">AI-powered question answering for your Slack workspace</p>
        </div>
        
        <div class="status">
            <strong>Ready for Installation!</strong><br>
            <strong>Features:</strong> Slash commands, RAG with Cohere, real-time indexing, source citations
        </div>
        
        <div class="feature">
            <h3>How it works:</h3>
            <ol>
                <li><strong>Install:</strong> Click "Add to Slack" to authorize the bot</li>
                <li><strong>Index:</strong> Bot automatically indexes recent messages from your channels</li>
                <li><strong>Ask:</strong> Mention the bot or use <code>/ask [your question]</code> to get AI-powered answers</li>
                <li><strong>Learn:</strong> Bot continuously learns from new messages</li>
            </ol>
        </div>
        
        <div class="demo-commands">
            <h3>Try these commands after installation:</h3>
            <h4>Bot Mentions (Recommended):</h4>
            <ul>
                <li><code>@YourBotName What did we decide about the pricing strategy?</code></li>
                <li><code>@YourBotName Who was working on the mobile app features?</code></li>
                <li><code>@YourBotName What are the main concerns about the new deployment?</code></li>
            </ul>
            <h4>Slash Commands (Alternative):</h4>
            <ul>
                <li><code>/ask What did we decide about the pricing strategy?</code></li>
                <li><code>/ask Who was working on the mobile app features?</code></li>
                <li><code>/ask What are the main concerns about the new deployment?</code></li>
            </ul>
        </div>
        
        <div style="text-align: center; margin: 40px 0;">
            <a href="/slack/install" class="btn">
                <img src="https://platform.slack-edge.com/img/add_to_slack.png" 
                     alt="Add to Slack" width="139" height="40">
            </a>
        </div>
        
        <div class="feature">
            <h3>Privacy & Security:</h3>
            <ul>
                <li>Messages are processed securely with proper Slack OAuth</li>
                <li>Data is isolated per workspace (multi-tenant architecture)</li>
                <li>Configurable channel allowlists and retention policies</li>
                <li>All requests are signature-verified by Slack</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def oauth_success_response(team_data: Dict[str, Any]) -> HTMLResponse:
    """OAuth success page after successful Slack app installation"""
    team_name = team_data.get('team', {}).get('name', 'your workspace')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Installation Success!</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 600px; 
                margin: 50px auto; 
                padding: 20px; 
                text-align: center;
            }}
            .success {{ 
                background: #e8f5e8; 
                padding: 20px; 
                border-radius: 8px; 
                border: 1px solid #4caf50;
                margin: 20px 0;
            }}
            .next-steps {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: left;
                margin: 20px 0;
            }}
            code {{
                background: #f1f3f4; 
                padding: 2px 6px; 
                border-radius: 3px; 
                font-family: 'Monaco', 'Menlo', monospace;
            }}
        </style>
    </head>
    <body>
        <h2>Successfully installed in {team_name}!</h2>
        
        <div class="success">
            <strong>Installation Complete</strong><br>
            The bot is now indexing your recent messages...
        </div>
        
        <div class="next-steps">
            <h3>Next Steps:</h3>
            <ol>
                <li>Go to your Slack workspace</li>
                <li>Wait ~30-60 seconds for message indexing</li>
                <li>Try mentioning the bot: <code>@YourBotName What have we been discussing lately?</code></li>
                <li>Or use slash commands: <code>/ask What did we decide about the project timeline?</code></li>
            </ol>
        </div>
        
        <p><em>Note: The bot will index messages from public channels you have access to.</em></p>
        
        <a href="/health" style="color: #4A154B;">View System Status</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def oauth_error_response(error_message: str) -> HTMLResponse:
    """OAuth error page when installation fails"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Installation Error</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 600px; 
                margin: 50px auto; 
                padding: 20px; 
                text-align: center;
            }}
            .error {{ 
                background: #ffebee; 
                padding: 20px; 
                border-radius: 8px; 
                border: 1px solid #f44336;
                margin: 20px 0;
            }}
            .btn {{
                background: #4A154B; 
                color: white; 
                padding: 15px 30px; 
                text-decoration: none; 
                border-radius: 6px; 
                display: inline-block;
                font-weight: 600;
                margin: 10px;
            }}
        </style>
    </head>
    <body>
        <h2>Installation Failed</h2>
        
        <div class="error">
            <strong>Error:</strong> {error_message}
        </div>
        
        <p>Please try installing the app again or contact support if the problem persists.</p>
        
        <a href="/" class="btn">Try Again</a>
        <a href="/health" class="btn">System Status</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=400)

def maintenance_page() -> HTMLResponse:
    """Maintenance page for when the system is down"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>System Maintenance</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 600px; 
                margin: 50px auto; 
                padding: 20px; 
                text-align: center;
            }
            .maintenance { 
                background: #fff3cd; 
                padding: 20px; 
                border-radius: 8px; 
                border: 1px solid #ffeaa7;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <h2>System Maintenance</h2>
        
        <div class="maintenance">
            <strong>We're currently performing maintenance</strong><br>
            The Slack RAG Bot is temporarily unavailable. Please check back in a few minutes.
        </div>
        
        <p>Thank you for your patience!</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=503)
