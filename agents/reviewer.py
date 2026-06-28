"""Quality Reviewer agent (the Critic).

Cross-references worker outputs against the raw transcript and emits a JSON
verdict. Note: there is NO ground-truth reference -- the reviewer judges
internal consistency / hallucination / structural defects against the
transcript only.

Expected verdict shape:
    {
      "status": "pass" | "revise",
      "issues": {
         "summarizer": "<critique or empty>",
         "extractor":  "<critique or empty>",
         "drafter":    "<critique or empty>"
      }
    }
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from agents import config
from agents.base import Agent, CallResult

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verdict:
    status: str                       # "pass" | "revise"
    issues: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def needs_revision(self) -> bool:
        return self.status == "revise" and any(v.strip() for v in self.issues.values())

    def workers_to_revise(self) -> list[str]:
        return [w for w, c in self.issues.items() if c and c.strip()]


def parse_verdict(text: str) -> Verdict:
    """Robustly parse the reviewer JSON, defaulting to 'pass' on garbage.

    Falling back to 'pass' on unparseable output silently drops the reviewer's
    judgement, so log a warning whenever non-empty text fails to parse (P3.3).
    Empty text (e.g. an errored reviewer call) is left silent — the run-level
    error guard (PipelineResult.finalize) already covers that case.
    """
    non_empty = bool((text or "").strip())
    match = _JSON_RE.search(text or "")
    if not match:
        if non_empty:
            logger.warning("reviewer verdict unparseable (no JSON object found); "
                           "defaulting to 'pass'. raw[:200]=%r", text[:200])
        return Verdict(status="pass", raw=text)
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("reviewer verdict JSON decode failed (%s); defaulting to "
                       "'pass'. raw[:200]=%r", exc, text[:200])
        return Verdict(status="pass", raw=text)
    status = str(obj.get("status", "pass")).lower()
    issues = obj.get("issues", {}) or {}
    if not isinstance(issues, dict):
        issues = {}
    issues = {k: str(v) for k, v in issues.items()}
    return Verdict(status=status, issues=issues, raw=text)


class Reviewer(Agent):
    def __init__(self, client: httpx.AsyncClient, **kw):
        super().__init__(config.AGENT_SPECS["reviewer"], client, **kw)

    async def run(self, transcript: str, summary: str, action_items: str,
                  followups: str, revision: int = 0) -> tuple[CallResult, Verdict]:
        user = (
            "Evaluate the three artifacts below strictly against the transcript. "
            "Return ONLY the JSON verdict described in your instructions.\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"SUMMARY:\n{summary}\n\n"
            f"ACTION_ITEMS:\n{action_items}\n\n"
            f"FOLLOWUPS:\n{followups}"
        )
        result = await self.call(user, revision=revision)
        return result, parse_verdict(result.text)
