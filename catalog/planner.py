"""Job planning: order a multi-model job to keep total cold-start time low (README "Job planning").

The plan is advice: the app runs the steps, taking one lease per step. Cold-start time is the only
cost weighed. Staged pipelines load every model once (the lowest possible total); co-resident
pipelines keep the slowest-loading stage in the outer loop. Only one GPU model runs at a time, so models
fit together only when at most one of them uses the GPU.
"""

from __future__ import annotations

from typing import Any

from tools.schema import Platform

from .index import OutputIndex
from .scheduler import Scheduler, SchedulerError


class PlanError(ValueError):
    pass


def plan(scheduler: Scheduler, index: OutputIndex, body: dict[str, Any]) -> dict[str, Any]:
    pipeline, stages = _check_body(body)
    p = scheduler.platform()
    stages = _order(stages)
    info = _model_info(scheduler, p, stages)
    stages = [{**st, "models": _distinct(st, info)} for st in stages]
    res = _Residency(info, index)
    if pipeline == "staged" or len(stages) == 1:
        steps = _staged_steps(stages, res)
    else:
        steps = _co_resident_steps(stages, res)
    return _summary(pipeline, steps, info)


def _summary(pipeline: str, steps: list[dict[str, Any]], info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The plan: its steps, how many loads they take and how long, the index hits, and whether the models fit
    together (at most one of them uses the GPU)."""
    loads = sum(1 for s in steps if s["load"])
    frac = {i["name"]: i["frac"] for i in info.values()}  # info is keyed by the ids asked for, steps by name
    fits = sum(frac[m] > 0 for m in {s["model"] for s in steps}) <= 1
    return {"pipeline": pipeline, "steps": steps, "loads": loads,
            "est_load_s": round(sum(s["cold_start_s"] for s in steps if s["load"]), 1),
            "hits": sum(s["hits"] or 0 for s in steps), "fits_together": fits}


def _check_body(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The pipeline and the stages of a plan request; PlanError when either is malformed."""
    pipeline = body.get("pipeline") or "staged"
    if pipeline not in ("staged", "co-resident"):
        raise PlanError("pipeline must be 'staged' or 'co-resident'")
    stages = body.get("stages") or []
    if not stages:
        raise PlanError("stages must not be empty")
    if not isinstance(stages, list) or not all(isinstance(st, dict) for st in stages):
        raise PlanError("stages must be a list of objects")
    for st in stages:
        _check_stage(st)
    return pipeline, stages


def _model_info(scheduler: Scheduler, p: Platform, stages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per model id asked for: its name, whether it runs, its cold start, its build key and its GPU share."""
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
    return info


class _Residency:
    """Which models are up as the plan's steps run (those running at plan time, until a GPU model loads), and
    how many of a step's index keys already have outputs."""

    def __init__(self, info: dict[str, dict[str, Any]], index: OutputIndex):
        self.info, self.index = info, index
        self.up = {i["name"] for i in info.values() if i["running"]}

    def loaded(self, m: str) -> bool:
        return self.info[m]["name"] in self.up

    def evict_for(self, m: str) -> None:
        """One GPU model runs at a time: loading m stops the GPU model that ran."""
        info = self.info
        if info[m]["frac"] > 0:
            self.up.difference_update({i["name"] for i in info.values() if i["frac"] > 0} - {info[m]["name"]})

    def counts(self, st: dict[str, Any], m: str) -> tuple[int | None, int | None, int | None]:
        """(items, hits, missing) of the stage's keys for m; all None when it gave none."""
        keys = (st.get("keys") or {}).get(m)
        if keys is None:
            return None, None, None
        bk = self.info[m]["build_key"]
        hits = self.index.count_hits([{**k, "build_key": bk} for k in keys]) if bk else 0
        return len(keys), hits, len(keys) - hits


def _staged_steps(stages: list[dict[str, Any]], res: _Residency) -> list[dict[str, Any]]:
    """One stage after another: in each, running models first (no load), then the quickest loads."""
    info = res.info
    steps: list[dict[str, Any]] = []
    for st in stages:
        for m in sorted(st["models"], key=lambda m: (not res.loaded(m), info[m]["cold"], m)):
            items, hits, missing = res.counts(st, m)
            skip = missing == 0
            load = not skip and not res.loaded(m)
            steps.append(_step(st["name"], m, info[m], items, hits, missing, skip, load=load))
            if load:
                res.evict_for(m)
    return steps


def _co_resident_steps(stages: list[dict[str, Any]], res: _Residency) -> list[dict[str, Any]]:
    """Two stages at once: the stage with the slowest load is the outer loop and stays up, and the inner stage
    runs alongside each of its models."""
    if len(stages) != 2:
        raise PlanError("a co-resident plan takes exactly two stages")
    info = res.info
    outer, inner = sorted(stages, key=lambda st: -max(info[m]["cold"] for m in st["models"]))
    steps: list[dict[str, Any]] = []
    loaded_inner: set[str] = set()
    for o in sorted(outer["models"], key=lambda m: (not res.loaded(m), info[m]["cold"], m)):
        items, hits, missing = res.counts(outer, o)
        load = missing != 0 and not res.loaded(o)
        steps.append(_step(outer["name"], o, info[o], items, hits, missing, missing == 0, load=load))
        if load:
            res.evict_for(o)
        all_fit = sum(info[m]["frac"] > 0 for m in [o, *inner["models"]]) <= 1  # one GPU model at a time
        steps += _inner_steps(inner, info[o]["name"], res, loaded_inner, all_fit)
    return steps


def _inner_steps(inner: dict[str, Any], alongside: str, res: _Residency, loaded_inner: set[str],
                 all_fit: bool) -> list[dict[str, Any]]:
    """The inner stage's steps next to one outer model: an inner model loaded before stays up if all fit."""
    info = res.info
    steps: list[dict[str, Any]] = []
    for i in sorted(inner["models"], key=lambda m: (info[m]["cold"], m)):
        items, hits, missing = res.counts(inner, i)
        stays = all_fit and i in loaded_inner
        load = missing != 0 and not res.loaded(i) and not stays
        steps.append(_step(inner["name"], i, info[i], items, hits, missing, missing == 0, load=load,
                           alongside=alongside))
        if load:
            res.evict_for(i)
        if load or res.loaded(i):
            loaded_inner.add(i)
    return steps


def _check_stage(st: dict[str, Any]) -> None:
    if not st.get("name") or not st.get("models"):
        raise PlanError("every stage needs a name and models")
    if not isinstance(st["name"], str) or not isinstance(st.get("after") or "", str):
        raise PlanError("a stage's name and after must be strings")
    if not isinstance(st["models"], list) or not all(isinstance(m, str) for m in st["models"]):
        raise PlanError(f"stage {st['name']!r}: models must be a list of model ids")
    keys = st.get("keys") or {}
    if not isinstance(keys, dict) or not all(v is None or (isinstance(v, list) and all(isinstance(k, dict) for k in v))
                                             for v in keys.values()):
        raise PlanError(f"stage {st['name']!r}: keys must map a model to a list of index keys")


def _distinct(st: dict[str, Any], info: dict[str, dict[str, Any]]) -> list[str]:
    """The stage's model ids, one per model (an alias and its id name the same model); an id given keys is kept."""
    keys = st.get("keys") or {}
    ids: dict[str, str] = {}
    for m in st["models"]:
        name = info[m]["name"]
        if name not in ids or (keys.get(ids[name]) is None and keys.get(m) is not None):
            ids[name] = m
    return list(ids.values())


def _step(stage: str, model: str, i: dict[str, Any], items, hits, missing, skip: bool, load: bool,
          alongside: str | None = None) -> dict[str, Any]:
    step = {"stage": stage, "model": i["name"], "load": load, "cold_start_s": round(i["cold"], 1),
            "cold_start_measured": i["measured"], "items": items, "hits": hits, "missing": missing, "skip": skip}
    if alongside:
        step["alongside"] = alongside
    return step


def _order(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stages in dependency order (`after`), stable otherwise."""
    names = {st["name"] for st in stages}
    for st in stages:
        if st.get("after") and st["after"] not in names:
            raise PlanError(f"stage {st['name']!r} runs after unknown stage {st['after']!r}")
    done: list[dict[str, Any]] = []
    left = list(stages)
    while left:
        ready = [st for st in left if not st.get("after") or st["after"] in {d["name"] for d in done}]
        if not ready:
            raise PlanError("stages depend on each other in a cycle")
        done.append(ready[0])
        left.remove(ready[0])
    return done
