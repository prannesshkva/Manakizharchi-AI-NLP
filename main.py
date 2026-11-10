import os
import uvicorn

from app.main import app  # noqa: F401 - Re-export ASGI app for `uvicorn main:app`


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
