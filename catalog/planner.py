"""Job planning: order a multi-model job to keep total cold-start time low (README "Job planning").

The plan is advice: the app runs the steps, taking one lease per step. Cold-start time is the only
cost weighed. Staged pipelines load every model once (the lowest possible total); co-resident
pipelines keep the slowest-loading stage in the outer loop.
"""

from __future__ import annotations

from typing import Any

from .index import OutputIndex
from .scheduler import Scheduler, SchedulerError


class PlanError(ValueError):
    pass


def plan(scheduler: Scheduler, index: OutputIndex, body: dict[str, Any]) -> dict[str, Any]:
    pipeline = body.get("pipeline") or "staged"
    if pipeline not in ("staged", "co-resident"):
        raise PlanError("pipeline must be 'staged' or 'co-resident'")
    stages = body.get("stages") or []
    if not stages:
        raise PlanError("stages must not be empty")
    p = scheduler.platform()
    names = {st.get("name") for st in stages}
    for st in stages:
        if not st.get("name") or not st.get("models"):
            raise PlanError("every stage needs a name and models")
        if st.get("after") and st["after"] not in names:
            raise PlanError(f"stage {st['name']!r} runs after unknown stage {st['after']!r}")
    stages = _order(stages)

    info: dict[str, dict[str, Any]] = {}
    for st in stages:
        for m in st["models"]:
            try:
                name = p.get(m).name
            except KeyError as exc:
                raise SchedulerError(404, exc.args[0]) from None
            view = scheduler.model_view(name)
            cold, measured = scheduler.cold_start(p, name)
            build = view["build"]
            info[m] = {"name": name, "running": view["state"] == "running", "cold": cold, "measured": measured,
                       "build_key": build.build_key if build else None, "frac": scheduler.frac(p, name)}

    def counts(st: dict[str, Any], m: str) -> tuple[int | None, int | None, int | None]:
        keys = (st.get("keys") or {}).get(m)
        if keys is None:
            return None, None, None
        bk = info[m]["build_key"]
        hits = index.count_hits([{**k, "build_key": bk} for k in keys]) if bk else 0
        return len(keys), hits, len(keys) - hits

    steps: list[dict[str, Any]] = []
    if pipeline == "staged" or len(stages) == 1:
        for st in stages:
            # Running models first (no load), then the quickest loads.
            for m in sorted(st["models"], key=lambda m: (not info[m]["running"], info[m]["cold"], m)):
                items, hits, missing = counts(st, m)
                skip = missing == 0
                steps.append(_step(st["name"], m, info[m], items, hits, missing, skip, load=not skip and not info[m]["running"]))
    else:
        # Co-resident: two stages at once. The stage with the slowest load is the outer loop and stays up.
        if len(stages) != 2:
            raise PlanError("a co-resident plan takes exactly two stages")
        outer, inner = sorted(stages, key=lambda st: -max(info[m]["cold"] for m in st["models"]))
        limit = p.config.util_limit
        loaded_inner: set[str] = set()
        for o in sorted(outer["models"], key=lambda m: (not info[m]["running"], info[m]["cold"], m)):
            items, hits, missing = counts(outer, o)
            steps.append(_step(outer["name"], o, info[o], items, hits, missing, missing == 0,
                               load=missing != 0 and not info[o]["running"]))
            all_fit = info[o]["frac"] + sum(info[i]["frac"] for i in inner["models"]) <= limit + 1e-9
            for i in sorted(inner["models"], key=lambda m: (info[m]["cold"], m)):
                items, hits, missing = counts(inner, i)
                stays = all_fit and i in loaded_inner
                load = missing != 0 and not info[i]["running"] and not stays
                steps.append(_step(inner["name"], i, info[i], items, hits, missing, missing == 0, load=load,
                                   alongside=o))
                if load or info[i]["running"]:
                    loaded_inner.add(i)

    loads = sum(1 for s in steps if s["load"])
    fits = sum(info[m]["frac"] for m in {s["model"] for s in steps}) <= p.config.util_limit + 1e-9
    return {"pipeline": pipeline, "steps": steps, "loads": loads,
            "est_load_s": round(sum(s["cold_start_s"] for s in steps if s["load"]), 1),
            "hits": sum(s["hits"] or 0 for s in steps), "fits_together": fits}


def _step(stage: str, model: str, i: dict[str, Any], items, hits, missing, skip: bool, load: bool,
          alongside: str | None = None) -> dict[str, Any]:
    step = {"stage": stage, "model": i["name"], "load": load, "cold_start_s": round(i["cold"], 1),
            "cold_start_measured": i["measured"], "items": items, "hits": hits, "missing": missing, "skip": skip}
    if alongside:
        step["alongside"] = alongside
    return step


def _order(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stages in dependency order (`after`), stable otherwise."""
    done: list[dict[str, Any]] = []
    left = list(stages)
    while left:
        ready = [st for st in left if not st.get("after") or st["after"] in {d["name"] for d in done}]
        if not ready:
            raise PlanError("stages depend on each other in a cycle")
        done.append(ready[0])
        left.remove(ready[0])
    return done
