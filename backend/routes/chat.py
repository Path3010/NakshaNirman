"""WebSocket chat route for real-time Groq-powered conversation."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, async_session
from models import Project
from services.chat import chat_with_groq
import json

router = APIRouter(tags=["chat"])


@router.websocket("/api/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with Groq."""
    await websocket.accept()

    history = []
    project_id = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            user_text = message.get("message", "")
            project_id = message.get("project_id", project_id)

            if not user_text:
                continue

            # Get response from Groq (or fallback)
            result = await chat_with_groq(user_text, history)

            # Update history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": result["reply"]})

            # Save history to project if we have one
            if project_id:
                try:
                    async with async_session() as db:
                        proj_result = await db.execute(
                            select(Project).where(Project.id == project_id)
                        )
                        project = proj_result.scalar_one_or_none()
                        if project:
                            project.chat_history = json.dumps(history)
                            await db.commit()
                except Exception:
                    pass  # Non-critical: don't break chat over DB issues

            # Send response
            await websocket.send_text(json.dumps({
                "reply": result["reply"],
                "extracted_data": result.get("extracted_data"),
                "should_generate": result.get("should_generate", False),
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "reply": f"Sorry, an error occurred: {str(e)}",
                "extracted_data": None,
                "should_generate": False,
            }))
        except Exception:
            pass
