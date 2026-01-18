"""
Simple Teams MCP Server - Send Messages Only
"""

import os
import requests
from fastmcp import FastMCP

mcp = FastMCP("Teams MCP")

WEBHOOK_URL = os.getenv(
    "TEAMS_WEBHOOK_URL",
    "https://ytpl.webhook.office.com/webhookb2/5a48e0c5-af76-4054-977a-3114e3e04dbc@2161a74d-1c3e-4d34-a8c8-131360d2e92c/IncomingWebhook/82828d6f61234b02854162f084719aec/02bbda5f-edda-494d-b05e-840cba8c182f/V2kZLcKbA9i6UPc_0BPP2ggaY9IE7yBmbx-rsERwVnfXw1",
)


@mcp.tool()
def send_message(message: str, title: str = "Message") -> dict:
    """Send a message to Teams channel.

    Args:
        message: Message text
        title: Message title
    """
    if not WEBHOOK_URL:
        return {"error": "Set TEAMS_WEBHOOK_URL in .env file"}

    payload = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "title": title,
        "text": message,
    }

    response = requests.post(WEBHOOK_URL, json=payload)

    if response.status_code == 200:
        return {"status": "sent", "message": message}
    return {"error": f"Failed: {response.status_code}"}


if __name__ == "__main__":
    mcp.run()
