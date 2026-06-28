"""Shared helpers + base class for worker agents (summarizer/extractor/drafter)."""
from __future__ import annotations

import httpx

from agents import config
from agents.base import Agent, CallResult, _load_prompt

_REVISION_PREAMBLE = (
    "A reviewer flagged the following issues with your previous output. "
    "Revise to address them:\n"
)


def compose_worker_input(brief: str, transcript: str, critique: str | None,
                         *, role_instruction: str | None = None,
                         shared_prefix: bool = False) -> str:
    """Build the worker prompt body.

    Default layout (``shared_prefix=False``): ``COORDINATION BRIEF`` then
    ``TRANSCRIPT``; the role-specific instruction lives in the *system* prompt.
    Note: because each worker's system prompt differs, vLLM cannot actually reuse
    the (brief+transcript) block across workers in this layout.

    Shared-prefix layout (``shared_prefix=True``, P1.3): ``TRANSCRIPT`` then
    ``COORDINATION BRIEF`` come FIRST and are identical across the three workers
    (paired with a common system preamble), so the long prefix is a genuine
    shared prefix that vLLM can dedupe from its KV cache; the role-specific
    ``TASK`` is appended at the END.
    """
    if shared_prefix:
        parts = [f"TRANSCRIPT:\n{transcript}", f"COORDINATION BRIEF:\n{brief}"]
        if role_instruction:
            parts.append(f"TASK:\n{role_instruction}")
        if critique:
            parts.append(_REVISION_PREAMBLE + critique)
        return "\n\n".join(parts)

    parts = [f"COORDINATION BRIEF:\n{brief}", f"TRANSCRIPT:\n{transcript}"]
    if critique:
        parts.append(_REVISION_PREAMBLE + critique)
    return "\n\n".join(parts)


class WorkerAgent(Agent):
    """Base for the three workers, handling both prompt layouts (P1.3).

    In shared-prefix mode the worker swaps its role-specific system prompt for the
    common worker preamble and relocates its role prompt to the trailing TASK
    block, so all workers share an identical leading prefix.
    """

    def __init__(self, spec: config.AgentSpec, client: httpx.AsyncClient, **kw):
        super().__init__(spec, client, **kw)
        self.role_instruction = self.system_prompt
        self.shared_prefix = config.SHARED_PREFIX_LAYOUT
        if self.shared_prefix:
            self.system_prompt = _load_prompt(config.WORKER_SHARED_PROMPT_FILE)

    async def run(self, brief: str, transcript: str,
                  critique: str | None = None, revision: int = 0) -> CallResult:
        body = compose_worker_input(
            brief, transcript, critique,
            role_instruction=self.role_instruction if self.shared_prefix else None,
            shared_prefix=self.shared_prefix)
        return await self.call(body, revision=revision)
