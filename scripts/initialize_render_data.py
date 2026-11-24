import os
import shutil
from pathlib import Path

SOURCE_PATH = Path("/app/data/processed/chroma_db")
TARGET_VALUE = os.getenv("CHROMA_PATH")

if not TARGET_VALUE:
    print("CHROMA_PATH is not configured. Using the image-local database.")
    raise SystemExit(0)

TARGET_PATH = Path(TARGET_VALUE)

if SOURCE_PATH.resolve() == TARGET_PATH.resolve():
    print("Using the existing image-local ChromaDB database.")
    raise SystemExit(0)

if not SOURCE_PATH.is_dir():
    raise RuntimeError(
        f"Prebuilt ChromaDB source not found: {SOURCE_PATH}"
    )

TARGET_PATH.mkdir(parents=True, exist_ok=True)

if any(TARGET_PATH.iterdir()):
    print(f"Existing ChromaDB data found at {TARGET_PATH}. Preserving it.")
else:
    print(f"Initializing ChromaDB at {TARGET_PATH}...")
    shutil.copytree(
        SOURCE_PATH,
        TARGET_PATH,
        dirs_exist_ok=True,
    )
    print("ChromaDB initialization completed.")

print("ChromaDB startup initialization finished.")
