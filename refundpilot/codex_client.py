"""Small structured-output client around an authenticated local Codex CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class CodexStructuredClient:
    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "low",
        timeout_seconds: int = 180,
        executable: Optional[str] = None,
    ) -> None:
        resolved = executable or shutil.which("codex")
        if resolved is None:
            raise RuntimeError("codex executable was not found on PATH")
        self.executable = resolved
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str, schema_path: Path) -> str:
        if not schema_path.exists():
            raise RuntimeError("structured output schema is missing: %s" % schema_path)

        with tempfile.TemporaryDirectory(prefix="refundpilot-codex-") as directory:
            output_path = Path(directory) / "response.json"
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                self.model,
                "-c",
                'model_reasoning_effort="%s"' % self.reasoning_effort,
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-",
            ]
            process_environment = os.environ.copy()
            process_environment["NO_COLOR"] = "1"
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=directory,
                    env=process_environment,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Codex structured call timed out") from exc
            if completed.returncode != 0:
                diagnostic = (completed.stderr or completed.stdout)[-1200:]
                raise RuntimeError(
                    "Codex structured call failed: %s" % diagnostic.strip()
                )
            if not output_path.exists():
                raise RuntimeError("Codex structured call produced no response")
            return output_path.read_text(encoding="utf-8")
