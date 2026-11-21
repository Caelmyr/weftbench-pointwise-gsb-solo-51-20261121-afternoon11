"""Start the local image workbench."""
import os
from pathlib import Path

from waitress import serve

from .web import create_app


def main():
    root = Path(os.environ.get("WEFTBENCH_DATA", "./data"))
    address = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app = create_app(root)
    print(f"WeftBench listening on http://{address}:{port}", flush=True)
    serve(app, host=address, port=port, threads=6, clear_untrusted_proxy_headers=True)


if __name__ == "__main__":
    main()
