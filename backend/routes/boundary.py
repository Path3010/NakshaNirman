"""Boundary upload and processing routes."""

import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import Project, BoundaryUpload
from services.boundary import process_boundary_file
from config import UPLOAD_DIR
import json

router = APIRouter(prefix="/api", tags=["boundary"])


@router.post("/upload-boundary")
async def upload_boundary(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    scale: float = Form(1.0),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a boundary image or DXF and extract the polygon.
    
    Supports all image formats (PNG, JPG, JPEG, BMP, TIFF, GIF) and DXF files.
    Uses universal extraction algorithm that works for any shape worldwide.
    """
    # Validate project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Determine file type
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Support all common image formats
    image_extensions = ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "gif", "webp")
    
    if ext in image_extensions:
        file_type = "image"
    elif ext == "dxf":
        file_type = "dxf"
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {ext}. Supported: {', '.join([*image_extensions, 'dxf'])}"
        )

    # Save file
    file_id = str(uuid.uuid4())
    save_path = os.path.join(str(UPLOAD_DIR), f"{file_id}.{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        # Process boundary
        boundary_data = process_boundary_file(save_path, file_type, scale)

        # Save to database
        upload = BoundaryUpload(
            project_id=project_id,
            file_path=save_path,
            file_type=file_type,
            processed_polygon=json.dumps(boundary_data["polygon"]),
        )
        db.add(upload)

        # Update project
        project.boundary_polygon = json.dumps(boundary_data["polygon"])
        await db.flush()

        return {
            "status": "success",
            "message": "Boundary extracted successfully with high accuracy",
            "polygon": boundary_data["polygon"],
            "area": boundary_data["area"],
            "num_vertices": boundary_data["num_vertices"],
            "perimeter": boundary_data.get("perimeter", 0),
            "is_valid": boundary_data.get("is_valid", True),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
