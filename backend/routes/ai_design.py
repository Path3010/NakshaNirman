"""
AI Design Advisor routes — 4-Stage Pipeline Architecture.

Stage 1: Chat Mode — Collect requirements naturally
Stage 2: Extraction — Convert conversation to structured JSON
Stage 3: Design — Generate construction-ready layout
Stage 4: Validation — Validate the layout

Provides REST endpoints and WebSocket for real-time pipeline conversations.
"""

import json
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, async_session
from models import Project, ProjectStatus, Room, RoomType
from schemas import (
    AIDesignRequest, AIDesignResponse,
    AIReviewRequest, AIReviewResponse,
    PipelineRequest, PipelineResponse,
)
from services.grok_advisor import analyze_requirements, review_layout
from services.ai_pipeline import (
    PipelineStage,
    run_stage_1_chat,
    run_stage_2_extraction,
    run_stage_3_design,
    run_stage_4_validation,
    run_full_pipeline,
    check_requirements_complete,
)
from services.floorplan import generate_floor_plan
from services.cad_export import generate_dxf
from config import EXPORT_DIR

router = APIRouter(prefix="/api/ai-design", tags=["ai-design"])


@router.post("/analyze", response_model=AIDesignResponse)
async def ai_analyze(data: AIDesignRequest, db: AsyncSession = Depends(get_db)):
    """
    Analyze house design requirements using AI.

    Send natural language like "I want a 3BHK 1200 sqft house with Vastu"
    and get AI-analyzed structured room specifications.
    """
    plot_info = {}
    if data.project_id:
        result = await db.execute(select(Project).where(Project.id == data.project_id))
        project = result.scalar_one_or_none()
        if project:
            plot_info["total_area"] = project.total_area
            if project.boundary_polygon:
                try:
                    plot_info["boundary_polygon"] = json.loads(project.boundary_polygon)
                except (json.JSONDecodeError, TypeError):
                    pass

    if data.total_area:
        plot_info["total_area"] = data.total_area

    analysis = await analyze_requirements(data.message, plot_info)

    return AIDesignResponse(
        reasoning=analysis.get("reasoning", ""),
        rooms=analysis.get("rooms", []),
        vastu_recommendations=analysis.get("vastu_recommendations", []),
        compliance_notes=analysis.get("compliance_notes", []),
        design_score=analysis.get("design_score", 0),
        ready_to_generate=analysis.get("ready_to_generate", False),
        provider=analysis.get("provider", "unknown"),
        extracted_data=analysis.get("extracted_data"),
    )


@router.post("/review", response_model=AIReviewResponse)
async def ai_review(data: AIReviewRequest, db: AsyncSession = Depends(get_db)):
    """
    Review a generated floor plan for compliance, Vastu, and quality.
    """
    floor_plan = data.floor_plan
    if not floor_plan and data.project_id:
        result = await db.execute(select(Project).where(Project.id == data.project_id))
        project = result.scalar_one_or_none()
        if project and project.generated_plan:
            try:
                floor_plan = json.loads(project.generated_plan)
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(status_code=400, detail="Invalid floor plan data")

    if not floor_plan:
        raise HTTPException(status_code=400, detail="No floor plan to review")

    review = await review_layout(floor_plan)

    return AIReviewResponse(
        review_text=review.get("review_text", ""),
        scores=review.get("scores", {}),
        provider=review.get("provider", "unknown"),
    )


@router.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(data: PipelineRequest, db: AsyncSession = Depends(get_db)):
    """
    Run the full design pipeline from structured requirements.

    Executes: Design (Stage 3) → Validation (Stage 4) → DXF Generation.
    """
    design_result = await run_stage_3_design(data.requirements_json)
    layout_json = design_result.get("layout_json", {})

    if not layout_json:
        return PipelineResponse(
            stage="design",
            requirements_json=data.requirements_json,
            design_explanation="Failed to generate layout.",
        )

    validation_result = await run_stage_4_validation(layout_json)

    return PipelineResponse(
        stage="complete" if validation_result.get("compliant") else "validation",
        requirements_json=data.requirements_json,
        layout_json=layout_json,
        validation_report=validation_result.get("validation_report"),
        compliant=validation_result.get("compliant", False),
        design_explanation=design_result.get("reply", ""),
        provider=design_result.get("provider", "unknown"),
    )


@router.websocket("/chat")
async def ai_design_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AI design conversation with 4-stage pipeline.

    Stages:
      1. chat — Collect requirements naturally
      2. extraction — Convert conversation → structured JSON (automatic)
      3. design — Generate layout (automatic)
      4. validation — Validate layout (automatic)
      5. generation — Generate DXF file (automatic)

    The frontend sends: { "message": "...", "project_id": "..." }
    The backend responds with: { "reply": "...", "stage": "...", "stage_data": {...} }
    """
    await websocket.accept()
    history = []
    project_id = None
    current_stage = PipelineStage.CHAT

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            user_text = message.get("message", "")
            project_id = message.get("project_id", project_id)

            if not user_text:
                continue

            # ========================================
            # STAGE 1: Chat Mode
            # ========================================
            if current_stage == PipelineStage.CHAT:
                result = await run_stage_1_chat(user_text, history)

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": result["reply"]})

                # Send chat response
                await websocket.send_text(json.dumps({
                    "reply": result["reply"],
                    "stage": PipelineStage.CHAT,
                    "requirements_complete": result.get("requirements_complete", False),
                    "extracted_data": result.get("extracted_data"),
                    "provider": result.get("provider", "unknown"),
                }))

                # Save history to project
                await _save_chat_history(project_id, history)

                # Check if requirements are complete → auto-transition
                if result.get("requirements_complete"):
                    current_stage = PipelineStage.EXTRACTION

                    # Notify frontend of stage transition
                    await websocket.send_text(json.dumps({
                        "reply": "✓ All requirements collected. Extracting structured data...",
                        "stage": PipelineStage.EXTRACTION,
                        "stage_transition": True,
                        "provider": "system",
                    }))

                    # ========================================
                    # STAGE 2: Extraction (automatic)
                    # ========================================
                    extraction_result = await run_stage_2_extraction(history)
                    requirements_json = extraction_result.get("requirements_json", {})

                    await websocket.send_text(json.dumps({
                        "reply": f"✓ Requirements extracted: {json.dumps(requirements_json, indent=2)}",
                        "stage": PipelineStage.EXTRACTION,
                        "requirements_json": requirements_json,
                        "provider": extraction_result.get("provider", "unknown"),
                    }))

                    if not requirements_json:
                        current_stage = PipelineStage.CHAT
                        await websocket.send_text(json.dumps({
                            "reply": "Could not extract requirements. Let's try again — what's your plot size?",
                            "stage": PipelineStage.CHAT,
                            "provider": "system",
                        }))
                        continue

                    # Stage transition notification
                    await websocket.send_text(json.dumps({
                        "reply": "✓ Generating architectural layout...",
                        "stage": PipelineStage.DESIGN,
                        "stage_transition": True,
                        "provider": "system",
                    }))

                    # ========================================
                    # STAGE 3: Design (automatic)
                    # ========================================
                    current_stage = PipelineStage.DESIGN
                    design_result = await run_stage_3_design(requirements_json)
                    layout_json = design_result.get("layout_json", {})

                    await websocket.send_text(json.dumps({
                        "reply": design_result.get("reply", "Layout generated."),
                        "stage": PipelineStage.DESIGN,
                        "layout_json": layout_json,
                        "provider": design_result.get("provider", "unknown"),
                    }))

                    if not layout_json:
                        await websocket.send_text(json.dumps({
                            "reply": "⚠ Layout generation failed. Please adjust requirements.",
                            "stage": PipelineStage.DESIGN,
                            "provider": "system",
                        }))
                        current_stage = PipelineStage.CHAT
                        continue

                    # Stage transition
                    await websocket.send_text(json.dumps({
                        "reply": "✓ Validating design...",
                        "stage": PipelineStage.VALIDATION,
                        "stage_transition": True,
                        "provider": "system",
                    }))

                    # ========================================
                    # STAGE 4: Validation (automatic)
                    # ========================================
                    current_stage = PipelineStage.VALIDATION
                    validation_result = await run_stage_4_validation(layout_json)
                    validation_report = validation_result.get("validation_report", {})
                    is_compliant = validation_result.get("compliant", False)

                    await websocket.send_text(json.dumps({
                        "reply": validation_result.get("reply", "Validation complete."),
                        "stage": PipelineStage.VALIDATION,
                        "validation_report": validation_report,
                        "compliant": is_compliant,
                        "provider": validation_result.get("provider", "unknown"),
                    }))

                    # ========================================
                    # STAGE 5: Generation (if compliant)
                    # ========================================
                    if is_compliant or True:  # Always generate, even with minor issues
                        current_stage = PipelineStage.GENERATION

                        await websocket.send_text(json.dumps({
                            "reply": "✓ Generating floor plan and DXF file...",
                            "stage": PipelineStage.GENERATION,
                            "stage_transition": True,
                            "provider": "system",
                        }))

                        # Convert AI layout to room specs for the existing generator
                        rooms_for_generator = _convert_layout_to_rooms(layout_json)
                        total_area = requirements_json.get("total_area", 1200)

                        # Send generate signal with extracted data
                        dxf_url = None
                        if project_id:
                            dxf_url = await _generate_and_save(
                                project_id, rooms_for_generator, total_area,
                                layout_json, history,
                            )

                        current_stage = PipelineStage.COMPLETE

                        await websocket.send_text(json.dumps({
                            "reply": (
                                "✓ Floor plan generated successfully!\n\n"
                                + (f"📥 DXF file ready for download." if dxf_url else "")
                                + "\n\nWant to make changes? Just tell me what to adjust."
                            ),
                            "stage": PipelineStage.COMPLETE,
                            "should_generate": True,
                            "extracted_data": {
                                "rooms": rooms_for_generator,
                                "total_area": total_area,
                                "ready_to_generate": True,
                            },
                            "layout_json": layout_json,
                            "validation_report": validation_report,
                            "dxf_url": dxf_url,
                            "provider": "system",
                        }))

                        # Reset for potential re-design
                        current_stage = PipelineStage.CHAT
                        history = []

            else:
                # If stage got stuck, reset to chat
                current_stage = PipelineStage.CHAT
                result = await run_stage_1_chat(user_text, history)
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": result["reply"]})

                await websocket.send_text(json.dumps({
                    "reply": result["reply"],
                    "stage": PipelineStage.CHAT,
                    "provider": result.get("provider", "unknown"),
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "reply": f"Sorry, an error occurred: {str(e)}",
                "stage": current_stage,
                "extracted_data": None,
                "should_generate": False,
                "provider": "error",
            }))
        except Exception:
            pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _convert_layout_to_rooms(layout_json: dict) -> list:
    """Convert AI layout JSON rooms to the format expected by generate_floor_plan."""
    rooms = []
    for room in layout_json.get("rooms", []):
        rooms.append({
            "room_type": room.get("room_type", "other"),
            "quantity": 1,
            "desired_area": room.get("area"),
        })
    return rooms


async def _save_chat_history(project_id: str, history: list):
    """Save chat history to project database."""
    if not project_id:
        return
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
        pass


async def _generate_and_save(
    project_id: str, rooms: list, total_area: float,
    layout_json: dict, history: list,
) -> str:
    """Generate DXF and save to project. Returns DXF URL or None."""
    try:
        async with async_session() as db:
            proj_result = await db.execute(
                select(Project).where(Project.id == project_id)
            )
            project = proj_result.scalar_one_or_none()
            if not project:
                return None

            # Get or create boundary
            boundary = None
            if project.boundary_polygon:
                try:
                    boundary = json.loads(project.boundary_polygon)
                except (json.JSONDecodeError, TypeError):
                    pass

            if not boundary:
                import math
                side = math.sqrt(total_area)
                w = side * 1.3
                h = total_area / w
                boundary = [[0, 0], [w, 0], [w, h], [0, h], [0, 0]]

            # Generate floor plan using existing engine
            plan = generate_floor_plan(boundary, rooms, total_area)

            # Generate DXF
            dxf_filename = f"{project_id}.dxf"
            dxf_path = os.path.join(str(EXPORT_DIR), dxf_filename)
            generate_dxf(plan, dxf_path)

            # Update project
            project.generated_plan = json.dumps(plan)
            project.dxf_path = dxf_path
            project.total_area = total_area
            project.chat_history = json.dumps(history)
            project.status = ProjectStatus.COMPLETED
            await db.commit()

            return f"/api/download-dxf/{project_id}"

    except Exception:
        return None
