"""Background sampler for system + server resource usage during a run.

Samples on a fixed interval in a daemon thread:
  - host CPU utilisation (%) and orchestrator-process RSS (MB) via psutil
  - vLLM gauges (KV-cache usage %, running/waiting request counts) via the
    metrics scraper -- these reveal whether "parallel" requests are actually
    being serialised by the server under KV-cache pressure.

Returns aggregate stats (mean / max / last) so the runner can attach them to a
run record. Compute/throughput is derived elsewhere from token counts.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import psutil

from bench.vllm_metrics import VllmMetricsScraper


def _agg(values: list[float]) -> dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "max": None, "last": None, "samples": 0}
    return {
        "mean": sum(vals) / len(vals),
        "max": max(vals),
        "last": vals[-1],
        "samples": len(vals),
    }


@dataclass
class ResourceSamples:
    cpu_percent: list[float] = field(default_factory=list)
    proc_rss_mb: list[float] = field(default_factory=list)
    sys_mem_used_mb: list[float] = field(default_factory=list)
    sys_mem_percent: list[float] = field(default_factory=list)
    gpu_cache_usage_perc: list[float] = field(default_factory=list)
    num_requests_running: list[float] = field(default_factory=list)
    num_requests_waiting: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "host_cpu_percent": _agg(self.cpu_percent),
            "orchestrator_rss_mb": _agg(self.proc_rss_mb),
            "host_mem_used_mb": _agg(self.sys_mem_used_mb),
            "host_mem_percent": _agg(self.sys_mem_percent),
            "vllm_gpu_cache_usage_perc": _agg(self.gpu_cache_usage_perc),
            "vllm_num_requests_running": _agg(self.num_requests_running),
            "vllm_num_requests_waiting": _agg(self.num_requests_waiting),
        }


class SystemSampler:
    def __init__(self, scraper: Optional[VllmMetricsScraper] = None,
                 interval_s: float = 0.25) -> None:
        self.scraper = scraper
        self.interval_s = interval_s
        self.samples = ResourceSamples()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process()

    def _loop(self) -> None:
        # Prime the SYSTEM-wide cpu_percent (the value we actually sample below);
        # priming the per-process reading here was a no-op since we never sample it.
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            try:
                self.samples.cpu_percent.append(psutil.cpu_percent(interval=None))
                self.samples.proc_rss_mb.append(self._proc.memory_info().rss / 1e6)
                vm = psutil.virtual_memory()
                self.samples.sys_mem_used_mb.append(vm.used / 1e6)
                self.samples.sys_mem_percent.append(vm.percent)
                if self.scraper is not None:
                    snap = self.scraper.snapshot()
                    if snap.available:
                        g = snap.gauges
                        if g.get("gpu_cache_usage_perc") is not None:
                            self.samples.gpu_cache_usage_perc.append(g["gpu_cache_usage_perc"])
                        if g.get("num_requests_running") is not None:
                            self.samples.num_requests_running.append(g["num_requests_running"])
                        if g.get("num_requests_waiting") is not None:
                            self.samples.num_requests_waiting.append(g["num_requests_waiting"])
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "SystemSampler":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
