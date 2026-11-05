"""Official-source model installation; resumable downloads and bounded extraction."""
from pathlib import Path
import hashlib
import json
import re
import ssl
import tarfile
import time
import urllib.request

PHOENIX_ROOT = "https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/"
PHOENIX_WAVE = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
PHOENIX_STAR = "lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
BOBCAT_URL = "https://zenodo.org/api/records/5063476/files/spectra_m+0.0.tar.gz/content"
BOBCAT_BYTES = 1204213079
BOBCAT_MD5 = "382d939520eb516e50cdfe98625232ec"


def digest_file(path, algorithm="sha256"):
    d = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def tls_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url, destination, expected_bytes, expected_md5=None):
    """Resume .part files only when Content-Range agrees; atomically finalize."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise ValueError("Existing file has an unexpected size; move it aside before retrying: " + str(destination))
        if expected_md5 and digest_file(destination, "md5") != expected_md5:
            raise ValueError("Existing archive fails the official checksum; move it aside before retrying: " + str(destination))
        print("Using complete cached file:", destination.name, flush=True)
        return destination
    part = destination.with_name(destination.name + ".part")
    tag_path = destination.with_name(destination.name + ".part.http.json")
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_bytes:
            raise ValueError("Partial file is larger than the expected download: " + str(part))
        if offset == expected_bytes:
            break
        headers = {"User-Agent": "Atmosphere-Lab/0.3 scientific-model-downloader", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes=%d-" % offset
            if tag_path.exists():
                previous = json.loads(tag_path.read_text())
                tag = previous.get("etag") or previous.get("last_modified")
                if tag:
                    headers["If-Range"] = tag
        print("Downloading %s (attempt %d, %.1f MB already saved)" % (destination.name, attempt+1, offset/1e6), flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60, context=tls_context()) as response:
                status = getattr(response, "status", response.getcode())
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match.group(1)) != offset or int(match.group(3)) != expected_bytes:
                        raise ValueError("Server returned an unexpected byte range: " + content_range)
                    file_mode = "ab"
                elif status == 200:
                    if offset:
                        print("Server did not resume; restarting this file safely.", flush=True)
                    offset, file_mode = 0, "wb"
                else:
                    raise ValueError("Unexpected HTTP status: %s" % status)
                tag_path.write_text(json.dumps({"etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}))
                received = offset
                last_progress = time.monotonic()
                with part.open(file_mode) as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        received += len(chunk)
                        if received > expected_bytes:
                            raise ValueError("Download is larger than its official expected size.")
                        if time.monotonic()-last_progress >= 3:
                            print("  %.1f / %.1f MB (%.1f%%)" % (received/1e6, expected_bytes/1e6, received/expected_bytes*100), flush=True)
                            last_progress = time.monotonic()
                if received != expected_bytes:
                    raise OSError("Connection ended at %d of %d bytes." % (received, expected_bytes))
            break
        except (OSError, TimeoutError) as exc:
            print("Download interrupted:", exc, flush=True)
            if attempt == 2:
                raise
            print("Retrying from saved progress…", flush=True)
    if not part.exists() or part.stat().st_size != expected_bytes:
        raise ValueError("Download did not produce the expected number of bytes.")
    if expected_md5:
        print("Checking the archive against its official MD5…", flush=True)
        if digest_file(part, "md5") != expected_md5:
            raise ValueError("Official archive checksum mismatch. Remove only this generated partial file and retry: " + str(part))
    part.replace(destination)
    if tag_path.exists():
        tag_path.unlink()
    print("Saved:", destination.name, flush=True)
    return destination


def extract_selected(archive, destinations):
    """Copy only explicitly requested regular members; never extract archive paths."""
    remaining = dict(destinations)
    selected = {}
    with tarfile.open(archive, "r|gz") as tar:
        for index, member in enumerate(tar):
            if index and index % 100 == 0:
                print("  Scanned %d archive entries…" % index, flush=True)
            if member.name not in remaining:
                continue
            if not member.isfile() or member.size > 50_000_000:
                raise ValueError("Requested spectrum is not a bounded regular file: " + member.name)
            target = Path(remaining.pop(member.name))
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name+".part")
            with tar.extractfile(member) as source, part.open("wb") as dest:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    dest.write(chunk)
            if part.stat().st_size != member.size:
                raise ValueError("Incomplete member extraction: " + member.name)
            # Avoid overwriting an unrelated existing model file.
            if target.exists() and digest_file(target) != digest_file(part):
                part.unlink()
                raise ValueError("Existing model differs from this archive; move it aside: " + str(target))
            part.replace(target)
            selected[member.name] = {"path": str(target), "bytes": member.size, "sha256": digest_file(target)}
            print("Selected:", member.name, flush=True)
            if not remaining:
                break
    if remaining:
        raise ValueError("Requested spectra absent from the official archive: " + ", ".join(remaining))
    return selected
