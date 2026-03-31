from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db
from ocr_agent import extract_document_data
from datetime import datetime

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

@router.post("/upload-doc")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: jpeg, png, webp, gif"
        )

    # Read image bytes
    image_bytes = await file.read()

    # Run OCR via OpenRouter
    try:
        extracted = extract_document_data(image_bytes, file.content_type)
    except Exception as e:
        err_msg = str(e)
        if "does not support image input" in err_msg:
            raise HTTPException(status_code=422, detail="Unable to process image. The AI model doesn't support this image format. Please try a different image.")
        raise HTTPException(status_code=422, detail=f"Failed to read document: {err_msg}")

    # Validate required fields
    required = ["date", "amount", "category", "description"]
    for field in required:
        if field not in extracted:
            raise HTTPException(status_code=422, detail=f"Missing field from OCR: {field}")

    # Parse and normalize the date to first day of its month (matches quarterly schema)
    try:
        parsed_date = datetime.strptime(extracted["date"], "%Y-%m-%d")
        month_date = parsed_date.replace(day=1).date()
    except ValueError:
        month_date = datetime.now().replace(day=1).date()

    # Insert into expenses table
    db.execute(text("""
        INSERT INTO expenses (company_id, amount, category, description, month)
        VALUES (1, :amount, :category, :description, :month)
    """), {
        "amount":      float(extracted["amount"]),
        "category":    str(extracted["category"]),
        "description": str(extracted["description"]),
        "month":       month_date,
    })
    db.commit()

    return {
        "success": True,
        "message": "Document processed and saved successfully",
        "data": {
            "date":        str(month_date),
            "amount":      float(extracted["amount"]),
            "category":    extracted["category"],
            "description": extracted["description"],
        }
    }
