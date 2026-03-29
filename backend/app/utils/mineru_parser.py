"""MinerU HTTP API wrapper: upload → poll → download → extract markdown."""

import io
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

from app.utils.logger import get_logger

log = get_logger(__name__)

# Default retry / timeout settings
_POLL_INTERVAL = 10   # seconds between status checks
_MAX_POLLS = 180      # 30 minutes max


class MinerUError(Exception):
    pass


class MinerUParser:
    def __init__(self, api_key: str, base_url: str, model_version: str = "vlm"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        # MinerU is a Chinese domestic service — bypass any system/VPN proxy
        self._session.proxies.update({"http": None, "https": None})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: Path) -> str:
        """Parse a PDF file and return its Markdown content."""
        log.info("mineru.parse.start", file=str(file_path))

        # Step 1: Get presigned upload URL
        upload_url, batch_id = self._request_upload(file_path)
        log.info("mineru.upload_url_obtained", batch_id=batch_id)

        # Step 2: Upload file bytes directly to OSS (no auth headers)
        # Parsing starts automatically once the file is uploaded — no separate trigger needed
        self._upload_to_oss(upload_url, file_path)
        log.info("mineru.file_uploaded", batch_id=batch_id)

        # Step 3: Poll until done
        result_url = self._poll_until_done(batch_id)
        log.info("mineru.parse_complete", batch_id=batch_id)

        # Step 5: Download ZIP and extract markdown
        markdown = self._download_and_extract(result_url)
        log.info("mineru.markdown_extracted", chars=len(markdown))
        return markdown

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_upload(self, file_path: Path) -> tuple[str, str]:
        """Request a presigned upload URL and batch_id from MinerU."""
        filename = file_path.name
        resp = self._session.post(
            f"{self.base_url}/file-urls/batch",
            json={
                "enable_formula": False,
                "enable_table": True,
                "layout_model": self.model_version,
                "files": [{"name": filename, "is_ocr": True, "data_id": filename}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise MinerUError(f"MinerU upload request failed: {data}")

        file_urls = data["data"]["file_urls"]
        if not file_urls:
            raise MinerUError("No file_urls returned from MinerU")

        batch_id = data["data"]["batch_id"]
        # MinerU v4 API returns file_urls[0] as a plain URL string, not {"url": "..."}
        entry = file_urls[0]
        upload_url = entry if isinstance(entry, str) else entry["url"]
        return upload_url, batch_id

    def _upload_to_oss(self, upload_url: str, file_path: Path) -> None:
        """PUT file bytes to OSS presigned URL (no extra headers)."""
        content = file_path.read_bytes()
        # No extra headers (OSS signed URL), no proxy (domestic service)
        resp = requests.put(upload_url, data=content, timeout=120, proxies={"http": None, "https": None})
        if resp.status_code not in (200, 201, 204):
            raise MinerUError(f"OSS upload failed: {resp.status_code} {resp.text[:200]}")

    def _poll_until_done(self, batch_id: str) -> str:
        """Poll batch status until done, return the result ZIP download URL."""
        for attempt in range(_MAX_POLLS):
            time.sleep(_POLL_INTERVAL)
            resp = self._session.get(
                f"{self.base_url}/extract-results/batch/{batch_id}",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 0:
                raise MinerUError(f"MinerU status check failed: {data}")

            results = data["data"].get("extract_result", [])
            if not results:
                log.debug("mineru.polling", attempt=attempt, status="no_result_yet")
                continue

            state = results[0].get("state", "")
            log.debug("mineru.polling", attempt=attempt, state=state)

            if state == "done":
                full_zip_url: Optional[str] = results[0].get("full_zip_url")
                if not full_zip_url:
                    raise MinerUError("MinerU done but no full_zip_url in response")
                return full_zip_url

            if state == "failed":
                raise MinerUError(f"MinerU parse failed for batch {batch_id}")

        raise MinerUError(f"MinerU polling timed out after {_MAX_POLLS * _POLL_INTERVAL}s")

    def _download_and_extract(self, zip_url: str) -> str:
        """Download the result ZIP and extract the full markdown content."""
        resp = requests.get(zip_url, timeout=120, proxies={"http": None, "https": None})
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find the full.md or any .md file
            md_files = [n for n in zf.namelist() if n.endswith(".md")]
            if not md_files:
                raise MinerUError("No markdown file found in MinerU result ZIP")

            # Prefer full.md
            target = next((n for n in md_files if "full" in n.lower()), md_files[0])
            return zf.read(target).decode("utf-8")
