import os
import io
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model_utils import get_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("petvision.api")

# Constants
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load PyTorch ResNet50 model once into CPU memory
    logger.info("Initializing PetVision classifier at application startup...")
    classifier = get_classifier()
    logger.info(f"PetVision initialized with {len(classifier.class_names)} classes.")
    yield
    # Teardown logic if needed
    logger.info("PetVision server shutting down.")


app = FastAPI(
    title="PetVision API",
    description="Dog and Cat Breed Classifier with PyTorch ResNet50 & Grad-CAM Explainable AI",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local testing & demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the single-page frontend application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="index.html not found in static directory"
        )
    return FileResponse(index_file, media_type="text/html")


@app.get("/health")
async def health_check():
    """Basic health check and model status."""
    classifier = get_classifier()
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "classes_count": len(classifier.class_names),
        "device": "cpu"
    }


@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    """
    Accepts an uploaded pet image (multipart/form-data),
    validates file extension and size (<= 5MB),
    runs ResNet50 inference, and generates Grad-CAM heatmap overlay.
    """
    # 1. Validate filename and extension
    filename = file.filename or "unknown"
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Please upload an image with .jpg, .jpeg, .png, or .webp format."
        )

    # 2. Validate MIME type if provided
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{content_type}'. Expected image/jpeg, image/png, or image/webp."
        )

    # 3. Read content with size enforcement (max 5 MB)
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the uploaded image file. Please try again."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty. Please select a valid pet photo."
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds the 5 MB limit ({len(content) / (1024 * 1024):.2f} MB provided). Please compress or choose a smaller image."
        )

    # 4. Open with Pillow to verify it is a valid, uncorrupted image
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()  # Verify image integrity
        # Re-open for processing since verify() alters the stream pointer
        image = Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid or readable image. Please ensure the file is not corrupted."
        )

    # 5. Run inference and Grad-CAM generation
    try:
        classifier = get_classifier()
        prediction_result = classifier.predict(image)
        return JSONResponse(content=prediction_result)
    except Exception as e:
        logger.error(f"Prediction pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the pet image. Please try again."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
