"""Writes openapi.yaml from the app definition:  python -m cli.export_openapi"""
from pathlib import Path

import yaml

from app.main import app

Path("openapi.yaml").write_text(yaml.safe_dump(app.openapi(), sort_keys=False, allow_unicode=True), encoding="utf-8")
print("openapi.yaml written")
