import json

def handler(event, context):
    """Minimal Python function to verify Netlify function detection."""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "ok": True,
            "name": "hello",
            "message": "Netlify Python functions are working",
        }),
    }
