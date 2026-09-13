from __future__ import annotations

import uvicorn

from aar.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "aar.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
