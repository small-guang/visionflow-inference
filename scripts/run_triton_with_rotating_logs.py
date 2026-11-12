#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
from typing import Optional


SIZE_UNITS = {
    "k": 1024,
    "kb": 1024,
    "m": 1024 * 1024,
    "mb": 1024 * 1024,
    "g": 1024 * 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
}


def parse_size(value: str) -> int:
    raw = value.strip().lower()
    if raw.isdigit():
        return int(raw)

    for suffix, multiplier in sorted(SIZE_UNITS.items(), key=lambda item: len(item[0]), reverse=True):
        if raw.endswith(suffix):
            number = raw[: -len(suffix)].strip()
            return int(float(number) * multiplier)

    raise argparse.ArgumentTypeError(f"invalid size: {value}")


class RotatingLogWriter:
    def __init__(self, path: str, max_bytes: int, backup_count: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = max(backup_count, 0)
        self.file: Optional[object] = None
        self.size = 0
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._open()

    def _open(self) -> None:
        self.file = open(self.path, "ab", buffering=0)
        self.size = self.file.tell()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def _rotate(self) -> None:
        self.close()

        if self.backup_count == 0:
            open(self.path, "wb").close()
            self._open()
            return

        oldest = f"{self.path}.{self.backup_count}"
        if os.path.exists(oldest):
            os.remove(oldest)

        for index in range(self.backup_count - 1, 0, -1):
            source = f"{self.path}.{index}"
            target = f"{self.path}.{index + 1}"
            if os.path.exists(source):
                os.replace(source, target)

        if os.path.exists(self.path):
            os.replace(self.path, f"{self.path}.1")

        self._open()

    def write(self, data: bytes) -> None:
        if not data:
            return

        if self.max_bytes <= 0:
            assert self.file is not None
            self.file.write(data)
            self.size += len(data)
            return

        offset = 0
        while offset < len(data):
            if self.size >= self.max_bytes:
                self._rotate()

            available = self.max_bytes - self.size
            chunk = data[offset : offset + available]
            assert self.file is not None
            self.file.write(chunk)
            self.size += len(chunk)
            offset += len(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and tee stdout/stderr to a size-rotated log file."
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("TRITON_LOG_FILE", "/var/log/triton/server.log"),
    )
    parser.add_argument(
        "--max-size",
        type=parse_size,
        default=parse_size(os.environ.get("TRITON_LOG_MAX_SIZE", "100M")),
        help="Maximum size for the active log file. Use 0 to disable rotation.",
    )
    parser.add_argument(
        "--backup-count",
        type=int,
        default=int(os.environ.get("TRITON_LOG_BACKUP_COUNT", "5")),
        help="Number of rotated files to keep.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("missing command after --")

    writer = RotatingLogWriter(args.log_file, args.max_size, args.backup_count)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    def forward_signal(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            writer.write(chunk)
    finally:
        writer.close()

    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
