"""Controller: scheduling + orchestration backend for a MiniMax-H3 GPU pool.

See docs/ARCHITECTURE.md for a component overview. Public HTTP surface is
built by ``controller.main.create_app()`` and served with uvicorn.
"""