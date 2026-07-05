"""Export the Trace pydantic model to a checked-in JSON Schema file under /schema.

Run after any change to ctx_capture.schema: python scripts/export_schema.py
"""

import json
from pathlib import Path

from ctx_capture.schema import SCHEMA_VERSION, Trace

OUT_DIR = Path(__file__).resolve().parent.parent / "schema"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"v{SCHEMA_VERSION}.json"
    out_path.write_text(json.dumps(Trace.model_json_schema(), indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
