from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import zipfile
import shutil
import uuid


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="ZIP Chunking Agent",
    description="Non-LLM FastAPI agent for ZIP file extraction and chunking",
    version="1.0.0"
)


# ============================================================
# Configuration
# ============================================================

UPLOAD_FOLDER = Path("uploads")

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# ============================================================
# Chunking Function
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================================
# Non-LLM Agent
# ============================================================

class ZipChunkingAgent:

    def __init__(
        self,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_zip(self, zip_path: str):

        # Create unique extraction directory
        extract_folder = (
            UPLOAD_FOLDER
            / f"extracted_{uuid.uuid4()}"
        )

        extract_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            # ------------------------------------------------
            # 1. Extract ZIP
            # ------------------------------------------------

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    extract_folder
                )

            # ------------------------------------------------
            # 2. Supported file types
            # ------------------------------------------------

            supported_extensions = {
                ".py",
                ".txt",
                ".md",
                ".json",
                ".xml",
                ".csv",
                ".java",
                ".c",
                ".cpp",
                ".h",
                ".html",
                ".css",
                ".js",
                ".sql",
                ".yaml",
                ".yml",
                ".properties",
                ".log",
                ".cob",
                ".cbl",
                ".cpy"
            }

            results = []

            files_processed = 0

            # ------------------------------------------------
            # 3. Find files
            # ------------------------------------------------

            for file_path in extract_folder.rglob("*"):

                if not file_path.is_file():
                    continue

                extension = (
                    file_path.suffix.lower()
                )

                if extension not in supported_extensions:
                    continue

                files_processed += 1

                # ------------------------------------------------
                # 4. Read file
                # ------------------------------------------------

                try:

                    text = file_path.read_text(
                        encoding="utf-8",
                        errors="ignore"
                    )

                except Exception as e:

                    results.append({
                        "file_name": str(
                            file_path.relative_to(
                                extract_folder
                            )
                        ),
                        "error": str(e)
                    })

                    continue

                # ------------------------------------------------
                # 5. Chunk file
                # ------------------------------------------------

                chunks = chunk_text(
                    text,
                    self.chunk_size,
                    self.overlap
                )

                # ------------------------------------------------
                # 6. Store chunks
                # ------------------------------------------------

                for chunk_id, chunk in enumerate(
                    chunks,
                    start=1
                ):

                    results.append({
                        "file_name": str(
                            file_path.relative_to(
                                extract_folder
                            )
                        ),
                        "file_type": extension,
                        "chunk_id": chunk_id,
                        "content": chunk
                    })

            # ------------------------------------------------
            # 7. Return result
            # ------------------------------------------------

            return {
                "files_processed": files_processed,
                "total_chunks": len(results),
                "chunks": results
            }

        finally:

            # ------------------------------------------------
            # Delete extracted files
            # ------------------------------------------------

            if extract_folder.exists():

                shutil.rmtree(
                    extract_folder,
                    ignore_errors=True
                )


# ============================================================
# Create Agent
# ============================================================

agent = ZipChunkingAgent(
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
)


# ============================================================
# Home API
# ============================================================

@app.get("/")
def home():

    return {
        "message": "ZIP Chunking Agent is running",
        "llm": False
    }


# ============================================================
# ZIP Processing API
# ============================================================

@app.post("/process-zip")
async def process_zip(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="File name is missing"
        )

    if not file.filename.lower().endswith(".zip"):

        raise HTTPException(
            status_code=400,
            detail="Only ZIP files are allowed"
        )

    # --------------------------------------------------------
    # Create unique ZIP filename
    # --------------------------------------------------------

    unique_filename = (
        f"{uuid.uuid4()}_{file.filename}"
    )

    zip_path = (
        UPLOAD_FOLDER
        / unique_filename
    )

    try:

        # ----------------------------------------------------
        # Save uploaded ZIP
        # ----------------------------------------------------

        with zip_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ----------------------------------------------------
        # Send ZIP to Agent
        # ----------------------------------------------------

        result = agent.process_zip(
            str(zip_path)
        )

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {
            "status": "success",
            "zip_file": file.filename,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            **result
        }

    except zipfile.BadZipFile:

        raise HTTPException(
            status_code=400,
            detail="Invalid ZIP file"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # ----------------------------------------------------
        # Delete uploaded ZIP
        # ----------------------------------------------------

        if zip_path.exists():

            zip_path.unlink()


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

