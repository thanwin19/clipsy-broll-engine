"""Output storage: S3-compatible (MinIO) or local disk. Returns download links."""
import mimetypes
import shutil
from pathlib import Path

from app.config import get_settings


class LocalStorage:
    name = "local"

    def __init__(self):
        s = get_settings()
        self.root = s.data_dir / "outputs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.base = s.public_base_url.rstrip("/")

    def put(self, local: Path, key: str) -> str:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        return self.url(key)

    def url(self, key: str) -> str:
        return f"{self.base}/files/{key}"

    def get(self, key: str, local: Path) -> Path:
        shutil.copyfile(self.root / key, local)
        return local

    def ok(self) -> bool:
        return self.root.exists()


class S3Storage:
    name = "s3"

    def __init__(self):
        import boto3
        from botocore.config import Config
        s = get_settings()
        self.s = s
        cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})
        mk = lambda ep: boto3.client("s3", endpoint_url=ep, aws_access_key_id=s.s3_access_key,
                                     aws_secret_access_key=s.s3_secret_key, region_name=s.s3_region, config=cfg)
        self.client = mk(s.s3_endpoint_url)
        self.signer = mk(s.s3_public_endpoint_url or s.s3_endpoint_url)
        self.bucket = s.s3_bucket
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def put(self, local: Path, key: str) -> str:
        ctype = mimetypes.guess_type(str(local))[0] or "application/octet-stream"
        self.client.upload_file(str(local), self.bucket, key, ExtraArgs={"ContentType": ctype})
        return self.url(key)

    def url(self, key: str) -> str:
        return self.signer.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key},
                                                  ExpiresIn=self.s.link_ttl_seconds)

    def get(self, key: str, local: Path) -> Path:
        self.client.download_file(self.bucket, key, str(local))
        return local

    def ok(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


_storage = None


def get_storage():
    global _storage
    if _storage is None:
        _storage = S3Storage() if get_settings().storage_backend == "s3" else LocalStorage()
    return _storage
