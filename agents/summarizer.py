"""Summarizer worker agent."""
from __future__ import annotations

import httpx

from agents import config
from agents.worker_util import WorkerAgent


class Summarizer(WorkerAgent):
    def __init__(self, client: httpx.AsyncClient, **kw):
        super().__init__(config.AGENT_SPECS["summarizer"], client, **kw)
