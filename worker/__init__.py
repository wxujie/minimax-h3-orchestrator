"""Worker-side package that runs inside a Kaggle notebook.

The worker exposes a small authenticated HTTP agent (``agent.py``) that the
controller reaches through a Cloudflare tunnel. The agent talks only to its own
local ComfyUI instance. This module imports from ``controller.constants`` /
``controller.workflow`` (bundled with the notebook) so the worker and the
controller share one definition of the workflow and states.
"""