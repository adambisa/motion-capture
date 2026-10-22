"""Read single files out of a huge remote zip with HTTP Range requests.

HaGRIDv2 512px is one 119 GB zip. We only want a few thousand images, so we
read the zip's table of contents (the central directory, at the end of the
file) once, then fetch each wanted member's bytes directly.
"""

import struct
import time
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass

HAGRID_512 = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"
)


def fetch(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Bytes start..end (inclusive), retrying flaky connections."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as res:
                if res.status != 206:
                    raise OSError(f"server ignored the Range header (HTTP {res.status})")
                data: bytes = res.read()
            if len(data) != end - start + 1:
                raise OSError("short read")
            return data
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as res:
        return int(res.headers["Content-Length"])


@dataclass(frozen=True, slots=True)
class Member:
    name: str
    offset: int  # of the local file header
    compressed: int
    method: int


class _Tail:
    """Just enough of a file object for zipfile to parse a central directory we downloaded."""

    def __init__(self, data: bytes, base: int, total: int) -> None:
        self.data, self.base, self.total, self.pos = data, base, total, 0

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.total + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        start = self.pos - self.base
        if start < 0:
            raise OSError("zipfile read outside the downloaded tail")
        chunk = self.data[start:] if n < 0 else self.data[start : start + n]
        self.pos += len(chunk)
        return chunk


def list_members(url: str, tail_bytes: int = 256 << 20) -> list[Member]:
    """The zip's table of contents. Downloads only the last `tail_bytes` of the file."""
    total = size(url)
    base = max(0, total - tail_bytes)
    tail = _Tail(fetch(url, base, total - 1), base, total)
    with zipfile.ZipFile(tail) as zf:  # type: ignore[arg-type]
        return [
            Member(i.filename, i.header_offset, i.compress_size, i.compress_type)
            for i in zf.infolist()
            if not i.is_dir()
        ]


def read_member(url: str, m: Member) -> bytes:
    """One member's contents: local header, then (stored or deflated) data."""
    head = fetch(url, m.offset, m.offset + 29)
    sig, *_rest = struct.unpack("<IHHHHHIIIHH", head)
    if sig != 0x04034B50:
        raise OSError(f"bad local header for {m.name}")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = m.offset + 30 + name_len + extra_len
    data = fetch(url, start, start + m.compressed - 1)
    if m.method == zipfile.ZIP_STORED:
        return data
    if m.method == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    raise OSError(f"unsupported compression {m.method} for {m.name}")
