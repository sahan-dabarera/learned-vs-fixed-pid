"""Interactive dashboard: the thesis controllers on one path and one disturbance.

Fixed PID, the PPO gain scheduler and direct PPO are run on a scenario set in
the drawer and overlaid on shared axes: trajectory, tracking error, the delay
and noise actually in force, the steering command, the wheel differential and
the scheduler gains.

    python -m pathfollow.dashboard                 # main controllers (thesis names)
    python -m pathfollow.dashboard --all-arms      # ablation controllers as well

Every episode is built as a real ``scenarios.Scenario`` and run through
``rollout.run_episode`` with the controller objects of the batch evaluator
(``evaluation.evaluate._controller``): same environment, same history length,
same gain range and rate limit read from the checkpoint record, deterministic
actions. ``tests/test_dashboard.py`` asserts that an interactive run equals the
evaluator row for row. The page is for looking at runs; it writes nothing, and
the numbers of the thesis come from ``data/results`` alone.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from pathfollow import config
from pathfollow.calibration import GainCalibration, PIDGains
from pathfollow.config import CONFIG, PATH_SPLITS
from pathfollow.core import paths
from pathfollow.env import PathFollowingEnv
from pathfollow.evaluation import evaluate
from pathfollow.evaluation.make_test_scenarios import _log_uniform, _schedule
from pathfollow.rollout import run_episode
from pathfollow.scenarios import CONDITIONS, DisturbanceEvent, Scenario


# --- palette ---------------------------------------------------------------

MUTED, ACCENT = "#93a1b8", "#4fc3f7"
PANEL, LINE = "#1a2233", "#2c3a57"
GOOD, BAD, WARN = "#66d9a3", "#ff7a7a", "#ffce6b"
PATH_COLOR = "#8b97ad"
# One per controller panel, in load order. Distinct enough to tell four
# trajectories apart on one corridor plot.
CONTROLLER_COLORS = ("#4fc3f7", "#c792ea", "#ffb26b", "#6bd6c4", "#ff8fa3", "#a3e635")

# The comparison this experiment is about is between three methods, not between
# whichever controllers happen to be selected. Colour is therefore a property of
# the method, so fixed PID is the same amber in every chart of every session and
# the eye can carry a reading from one figure to the next. Seeds and regimes
# within a method get successive shades of that method's hue.
METHOD_COLORS = {
    "pid": ("#ffb26b", "#e08a3c", "#ffd7a8", "#b56c22"),
    "scheduled": ("#4fc3f7", "#2d8fbf", "#a5e4ff", "#1d6a90"),
    "direct": ("#c792ea", "#9a63c0", "#e3c6f7", "#6f3f93"),
}
METHOD_LABELS = {
    "pid": "Fixed PID",
    "scheduled": "PPO gain scheduler",
    "direct": "Direct PPO",
}
METHOD_ORDER = ("pid", "scheduled", "direct")

# Kept clear of every METHOD_COLORS hue: the disturbance bands are drawn on top
# of the method traces, and a band the colour of one of the methods reads as
# that method's own shading.
DELAY_TINT, NOISE_TINT = "#ff8fa3", "#6bd6c4"
EVENT_TINTS = {"delay_step": DELAY_TINT, "noise_step": NOISE_TINT}
EVENT_LABELS = {"delay_step": "dead time step", "noise_step": "sensor noise step"}

CORRIDOR_M = float(CONFIG.corridor_m)
CORRIDOR_GRID = 240
CORRIDOR_STRIDE = 5

SAFE_ARM = re.compile(r"^[a-z0-9_]+$")

DEFAULT_MODELS_ROOT = config.CHECKPOINTS
DEFAULT_CALIBRATION = config.CALIBRATION / "calibration.json"
DEFAULT_SCHEDULER_CALIBRATION = config.CALIBRATION / "blind_ppo.json"

# Thesis name of every trained arm. The first four are the main learned
# controllers and are always offered; the others are the ablation controllers,
# offered with --all-arms.
MAIN_ARMS = {
    "ppo_scheduler_ranged_h1_wide_2m": "GS-D",
    "ppo_scheduler_nominal_h1_wide": "GS-N",
    "ppo_direct_ranged_h1": "DR-D",
    "ppo_direct_nominal_h1": "DR-N",
}
ABLATION_ARMS = {
    "ppo_scheduler_ranged_h1_wide": "GS-D (300k)",
    "ppo_scheduler_ranged_h10_wide": "GS-D H10",
    "ppo_scheduler_ranged_h1": "narrow-range GS-D (300k)",
    "ppo_scheduler_ranged_h1_2m": "narrow-range GS-D (2M)",
    "ppo_scheduler_nominal_h1": "narrow-range GS-N",
    "ppo_scheduler_ranged_h10": "narrow-range GS-D H10",
    "ppo_scheduler_disturbed_h1": "point-trained GS-D",
    "ppo_scheduler_disturbed": "point-trained GS-D H10",
    "ppo_scheduler_nominal": "narrow-range GS-N H10",
    "ppo_direct_ranged_h10": "DR-D H10",
    "ppo_direct_disturbed_h1": "point-trained DR-D",
    "ppo_direct_disturbed": "point-trained DR-D H10",
    "ppo_direct_nominal": "DR-N H10",
}
ARM_NAMES = {**MAIN_ARMS, **ABLATION_ARMS}
# Seeds whose clean-run error is closest to the ten-seed median (stage 5 report).
REPRESENTATIVE = ("ppo_scheduler_ranged_h1_wide_2m:53", "ppo_direct_ranged_h1:37")

# The three fixed PIDs of the thesis: (name, Kp, Ki, Kd, tau_d or None). PID-N and
# PID-R are read from the calibration file; PID-RN is the selected point of the
# noise-aware grid (data/results/stage2/pid_retune_wide_fine/pid_rn.json).
PID_RN = (2.0, 0.5, 0.5, 0.01)
PID_LABELS = {
    "pid_global_robust_noise": "PID-RN",
    "pid_global_robust": "PID-R",
    "pid_global_nominal": "PID-N",
    "pid_custom": "hand-tuned",
}
# Bounds of the hand-tuned gain sliders: wide enough for all three fixed PIDs.
CUSTOM_GAIN_BOUNDS = ((1.0, 300.0), (0.0, 5.0), (0.0, 6.0))
EVALUATION_MODES = ("stationary", "transient", "dynamic")
DYNAMIC_NOISE_FLOOR_M = 0.0001  # lower end of the noise range, 0.1 mm

# The external conditions, and only those.
#
# Named because the slider and the validator behind it must agree. They did not:
# the slider offered 300 ms of dead time and the validator rejected anything over
# 200, so the top third of the track produced an error instead of an episode.
#
# Neither ceiling is a plant limit; they are the range worth exploring. The
# delay sweep of the thesis ends at 0.45 s and its noise sweep at 100 mm.
DELAY_MAX_MS = 500.0
NOISE_MAX_MM = 100.0
EVENT_START_MAX_S = 60.0

# (id, label, min, max, step, default)
SCENARIO_CONTROLS = (
    ("delay", "Dead time (ms)", 0.0, DELAY_MAX_MS, 1.0, 1000.0 * CONFIG.delay_severity_s),
    ("noise", "Sensor noise (mm)", 0.0, NOISE_MAX_MM, 0.01, 1000.0 * CONFIG.noise_severity_m),
    ("event-time", "Starts at (s)", 0.0, EVENT_START_MAX_S, 0.5, CONFIG.transient_start_s),
    ("seed", "Noise seed", 0.0, 100000.0, 1.0, 20260819.0 % 100000),
)
GAIN_LABELS = ("Kp", "Ki", "Kd")
GAIN_STEPS = (0.1, 0.01, 0.05)

# Metrics shown in the compact header row, in this order. Everything else in
# `metrics` still appears in the full table underneath.
HEADLINE_METRICS = (
    ("finished", "Completed"),
    ("failure_adjusted_error_m", "E_FA"),
    ("mean_distance_m", "Mean distance"),
    ("max_distance_m", "Max distance"),
    ("progress_pct", "Progress"),
    ("duration_s", "Duration"),
)


# --- small helpers ---------------------------------------------------------


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return parsed


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _allocation_spans(rows: list[dict[str, Any]]) -> list[list[float]]:
    """Contiguous stretches where the allocator clipped a wheel command.

    Derived rather than read from the trace: `env.py` computes the same test
    (`u_applied != u_requested`) but only records the fraction in the episode
    metrics, and the dashboard must not change the trace schema the evaluator
    writes.

    Merged into runs because an episode is ~1000 control steps; one plotly
    shape per sample is a chart that takes seconds to draw and says nothing
    more than a handful of spans do.
    """
    times = _values(rows, "t")
    if not times:
        return []
    step = (times[-1] - times[0]) / max(len(times) - 1, 1)
    spans: list[list[float]] = []
    for row in rows:
        try:
            moment = float(row["t"])
            limited = not (
                math.isclose(float(row["u_left"]), float(row["u_left_applied"]), abs_tol=1e-9)
                and math.isclose(float(row["u_right"]), float(row["u_right_applied"]), abs_tol=1e-9)
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not limited:
            continue
        if spans and moment - spans[-1][1] <= step * 1.5:
            spans[-1][1] = moment + step
        else:
            spans.append([moment, moment + step])
    return spans


def path_label(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    params = spec.get("params", {})
    if kind == "random_curvature":
        return f"random curvature (seed {params.get('seed', '?')})"
    return kind


def path_value(spec: dict[str, Any]) -> str:
    """A JSON string, so a whole path spec survives a dropdown round trip."""
    return json.dumps(spec, sort_keys=True)


def resolve_calibration(explicit: str | Path | None = None) -> tuple[GainCalibration, str]:
    """The calibration artifact, or the development default with a warning.

    A missing calibration is fatal in the batch evaluator and must stay that
    way. Here it is only degraded: the interactive tab is useful for hand
    tuning against the live plant before `calibrate_pid.py` has ever run, and
    refusing to start would remove the tool exactly when it is most wanted. The
    banner says which one is loaded so no one mistakes a smoke-test gain box
    for the calibrated one.
    """
    candidate = Path(explicit) if explicit is not None else DEFAULT_CALIBRATION
    if candidate.exists():
        if candidate.suffix.lower() != ".json":
            raise ValueError(f"calibration is not a JSON artifact: {candidate}")
        return GainCalibration.load(candidate.resolve(strict=True)), candidate.name
    if explicit is not None:
        raise FileNotFoundError(f"no calibration at {candidate}")
    return GainCalibration.development_default(), "development default (NOT calibrated)"


def discover_models(root: Path, all_arms: bool = True) -> list[dict[str, Any]]:
    """Every `<arm>/seed_<n>/best_model.zip` under the models root.

    The arm name carries the mode, and `metadata.json` states it outright; both
    are checked because loading a scheduler artifact into a direct-mode
    environment silently reinterprets the action vector.
    """
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    order = list(ARM_NAMES)
    for arm_dir in sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name),
    ):
        arm = arm_dir.name
        if not all_arms and arm not in MAIN_ARMS:
            continue
        if not SAFE_ARM.fullmatch(arm) or not arm.startswith(("ppo_scheduler_", "ppo_direct_")):
            continue
        for seed_dir in sorted(arm_dir.glob("seed_*"), key=lambda p: (len(p.name), p.name)):
            model_path = seed_dir / "best_model.zip"
            if not model_path.exists():
                continue
            declared = arm.split("_")[1]
            mode = "scheduled" if declared == "scheduler" else "direct"
            metadata_path = seed_dir / "metadata.json"
            if metadata_path.exists():
                recorded = json.loads(metadata_path.read_text(encoding="utf-8")).get("mode")
                if recorded and recorded != mode:
                    raise ValueError(
                        f"{model_path} sits under arm '{arm}' but its metadata records "
                        f"mode '{recorded}'. The artifact and its directory disagree."
                    )
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            found.append(
                {
                    "key": f"{arm}:{seed}",
                    # Short inside its method group, where the method is the
                    # heading; the panels and legends prepend the method name.
                    "label": f"{ARM_NAMES.get(arm, arm)} · seed {seed}",
                    "arm": arm,
                    "seed": seed,
                    "mode": mode,
                    "path": model_path.resolve(),
                }
            )
    return found


# --- the runner ------------------------------------------------------------


class InteractiveRunner:
    """Build scenarios, run controllers on them, cache the results.

    One instance per app. PPO artifacts are loaded once and kept; episodes are
    cached on (scenario, controller, gains) so that dragging a gain slider
    re-runs only the PID panel and leaves every policy panel untouched --
    otherwise the comparison would silently be against a different episode.
    """

    def __init__(
        self,
        models_root: Path | None = None,
        calibration_path: str | Path | None = None,
        scheduler_calibration_path: str | Path | None = None,
        all_arms: bool = False,
    ) -> None:
        self.calibration, self.calibration_label = resolve_calibration(calibration_path)
        self.calibration_path = Path(calibration_path or DEFAULT_CALIBRATION).resolve()
        self.scheduler_calibration_path = Path(
            scheduler_calibration_path or DEFAULT_SCHEDULER_CALIBRATION
        ).resolve()
        self.scheduler_calibration = GainCalibration.load(self.scheduler_calibration_path)
        self.models_root = Path(models_root or DEFAULT_MODELS_ROOT)
        self.all_arms = all_arms
        self.models = discover_models(self.models_root, all_arms)
        self.model_error = (
            ""
            if self.models
            else f"No trained checkpoint found under {self.models_root}."
        )
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._lock = Lock()

    # -- controllers --

    def controller_options(self) -> dict[str, list[dict[str, str]]]:
        """Selectable controllers, grouped by the three compared methods.

        Grouped rather than one flat list because the flat list made a
        twenty-three-entry checklist in which the three-way comparison the
        experiment is about was something you had to reconstruct by reading
        arm-name prefixes.
        """
        nominal, robust = self.calibration.nominal, self.calibration.robust
        grouped: dict[str, list[dict[str, str]]] = {
            "pid": [
                {"label": f"PID-RN (Kp {PID_RN[0]:g})", "value": "pid_global_robust_noise"},
                {"label": f"PID-R (Kp {robust.kp:g})", "value": "pid_global_robust"},
                {"label": f"PID-N (Kp {nominal.kp:g})", "value": "pid_global_nominal"},
                {"label": "hand-tuned (advanced sliders)", "value": "pid_custom"},
            ],
            "scheduled": [],
            "direct": [],
        }
        for entry in self.models:
            grouped[method_of(entry["key"])].append(
                {"label": entry["label"], "value": entry["key"]}
            )
        return grouped

    def default_selection(self) -> dict[str, list[str]]:
        """One controller per method, so the first run is already a comparison.

        PID-RN, GS-D and DR-D: the strongest fixed PID against the two
        range-trained learned controllers, each with its representative seed.
        """
        grouped = self.controller_options()
        selection = {"pid": ["pid_global_robust_noise"], "scheduled": [], "direct": []}
        for method in ("scheduled", "direct"):
            values = [option["value"] for option in grouped[method]]
            preferred = [key for key in REPRESENTATIVE if key in values and method_of(key) == method]
            if preferred or values:
                selection[method] = [(preferred or values)[0]]
        return selection

    def _model_entry(self, key: str) -> dict[str, Any]:
        for entry in self.models:
            if entry["key"] == key:
                return entry
        raise ValueError(f"controller '{key}' is no longer loaded")

    def controller(self, key: str, custom: PIDGains) -> dict[str, Any]:
        """Environment and policy of one controller, built as the evaluator builds them.

        Returns the environment, the predictor (None for a fixed PID), a label,
        a note, the gain range the controller works in, and whether the caller
        must close the environment (only the hand-tuned PID, which is not cached).
        """
        calibration = str(self.calibration_path)
        if key == "pid_custom":
            env = PathFollowingEnv(
                mode="fixed", training=False, calibration=self.calibration, fixed_gains=custom
            )
            return {"env": env, "predictor": None, "label": PID_LABELS[key], "close": True,
                    "note": f"Kp {custom.kp:.3f}, Ki {custom.ki:.3f}, Kd {custom.kd:.3f}",
                    "calibration": self.calibration}
        if key.startswith("pid_"):
            if key not in PID_LABELS:
                raise ValueError(f"unknown fixed PID: {key}")
            tau = None
            if key == "pid_global_robust_noise":
                gains, tau = PID_RN[:3], PID_RN[3]
            else:
                source = self.calibration.nominal if key == "pid_global_nominal" else self.calibration.robust
                gains = tuple(float(value) for value in source.as_array())
            spec = evaluate.ControllerSpec(key, None, "fixed", calibration, gains, derivative_filter_tau_s=tau)
            env, predictor = evaluate._controller(spec)
            note = f"Kp {gains[0]:.3f}, Ki {gains[1]:.3f}, Kd {gains[2]:.3f}"
            if tau is not None:
                note += f", derivative filter {tau:g} s"
            return {"env": env, "predictor": predictor, "label": PID_LABELS[key], "close": False,
                    "note": note, "calibration": self.calibration}
        entry = self._model_entry(key)
        spec = evaluate._scheduler_spec(
            entry["arm"], entry["seed"], entry["path"], calibration, str(self.scheduler_calibration_path)
        )
        env, predictor = evaluate._controller(spec)
        box = GainCalibration.load(Path(spec.calibration_path))
        note = f"{entry['arm']} · seed {entry['seed']} · {env.history_length}-frame input"
        if entry["mode"] == "scheduled":
            note += f" · gain range Kp {box.lower.kp:g}–{box.upper.kp:g}, Kd {box.lower.kd:g}–{box.upper.kd:g}"
        return {"env": env, "predictor": predictor, "label": entry["label"], "close": False,
                "note": note, "calibration": box}

    # -- scenarios --

    def scenario_from_controls(
        self,
        path_spec_json: str,
        target_speed: Any,
        condition: str,
        evaluation_mode: str,
        delay_ms: Any,
        noise_mm: Any,
        event_time: Any,
        noise_seed: Any,
    ) -> Scenario:
        """A Scenario assembled exactly the way `build_manifest` assembles one.

        Same field-for-field construction, so a hand-built scenario that
        happens to match a manifest entry runs the identical episode. The only
        difference is the id, which is prefixed `interactive-` so nothing
        downstream can mistake it for a manifest scenario.
        """
        try:
            spec = json.loads(path_spec_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("select a valid path") from error
        if not isinstance(spec, dict) or not isinstance(spec.get("kind"), str):
            raise ValueError("select a valid path")
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        if evaluation_mode not in EVALUATION_MODES:
            raise ValueError("evaluation mode must be stationary, transient or dynamic")
        speed = _number(target_speed, "target speed", 0.05, 5.0)

        delay_s = 0.001 * _number(delay_ms, "dead time", 0.0, DELAY_MAX_MS)
        noise_m = 0.001 * _number(noise_mm, "sensor noise", 0.0, NOISE_MAX_MM)
        start_s = _number(event_time, "transient start", 0.0, EVENT_START_MAX_S)
        # Bounded by the field, not by the slider: manifest seeds are
        # 20260819 + index, far outside the slider's convenience range, and
        # a scenario copied out of a manifest has to be reproducible here.
        seed = int(_number(noise_seed, "noise seed", 0.0, 2**31 - 1))

        wants_delay = condition in {"delay", "combined"}
        wants_noise = condition in {"noise", "combined"}
        stationary = evaluation_mode == "stationary"
        dynamic = evaluation_mode == "dynamic"

        built = paths.build(spec)
        geometry = (
            paths.validate_geometry(built)
            if spec["kind"] == "random_curvature"
            else paths.geometry(built)
        )
        initial_delay = delay_s if (stationary and wants_delay) else 0.0
        initial_noise = noise_m if (stationary and wants_noise) else 0.0
        events: list[DisturbanceEvent] = []
        if dynamic:
            # The dynamic-delay test of the thesis: the delay is drawn uniform in
            # [0, slider value] and redrawn after holds of 1 to 3 s; noise, when
            # the condition has it, is redrawn log-uniform in [0.1 mm, slider
            # value] on its own schedule. The noise seed fixes both schedules.
            horizon = (
                CONFIG.max_time_factor * float(geometry["length_m"]) / speed + CONFIG.max_time_margin_s
            )
            if wants_delay:
                initial_delay, changes = _schedule(
                    np.random.default_rng([seed, 0]), horizon,
                    lambda rng: float(rng.uniform(0.0, delay_s)),
                )
                events += [DisturbanceEvent("delay_step", start, value) for start, value in changes]
            if wants_noise:
                low = min(DYNAMIC_NOISE_FLOOR_M, noise_m)
                draw = (lambda rng: _log_uniform(rng, (low, noise_m))) if noise_m > 0.0 else (lambda rng: 0.0)
                initial_noise, changes = _schedule(np.random.default_rng([seed, 1]), horizon, draw)
                events += [DisturbanceEvent("noise_step", start, value) for start, value in changes]
            events.sort(key=lambda event: event.start_s)
        elif not stationary:
            if wants_delay:
                events.append(DisturbanceEvent("delay_step", start_s, delay_s))
            if wants_noise:
                events.append(DisturbanceEvent("noise_step", start_s, noise_m))

        return Scenario(
            scenario_id=f"interactive-{_slugify(path_label(spec))}-v{speed:.1f}-{evaluation_mode}-{condition}",
            path=dict(spec),
            target_speed=float(speed),
            # A dynamic scenario is stored as the thesis stores it: a stationary
            # record whose delay and noise change through its events.
            evaluation_mode="stationary" if dynamic else evaluation_mode,
            condition=condition,
            noise_seed=seed,
            initial_delay_s=float(initial_delay),
            initial_noise_std_m=float(initial_noise),
            events=tuple(events),
            geometry=geometry,
        )

    # -- episodes --

    def run(
        self,
        scenario: Scenario,
        controller_keys: Iterable[str],
        gains: tuple[float, float, float],
    ) -> list[dict[str, Any]]:
        keys = [key for key in controller_keys if key]
        if not keys:
            raise ValueError("select at least one controller")
        custom = PIDGains.from_iterable(gains)
        lower = np.array([bound[0] for bound in CUSTOM_GAIN_BOUNDS])
        upper = np.array([bound[1] for bound in CUSTOM_GAIN_BOUNDS])
        if np.any(custom.as_array() < lower) or np.any(custom.as_array() > upper):
            raise ValueError("drawer gains are outside the range of the gain sliders")

        # Method order, not selection order: the legend then reads PID, then
        # scheduled, then direct in every chart regardless of the order the
        # checkboxes were ticked in.
        keys.sort(key=lambda key: METHOD_ORDER.index(method_of(key)))
        scenario_key = json.dumps(asdict(scenario), sort_keys=True)
        used: dict[str, int] = {}
        results: list[dict[str, Any]] = []
        with self._lock:
            for key in keys:
                # Only `pid_custom` depends on the gain sliders; keying the
                # others without the gains is what keeps their episodes stable
                # while the sliders move.
                cache_key = (scenario_key, key) + (
                    tuple(custom.as_array().tolist()) if key == "pid_custom" else ()
                )
                result = self._cache.get(cache_key)
                if result is None:
                    result = self._run_one(scenario, key, custom)
                    self._cache[cache_key] = result
                method = method_of(key)
                shades = METHOD_COLORS[method]
                shade = used.get(method, 0)
                used[method] = shade + 1
                results.append(
                    {
                        **result,
                        "method": method,
                        "label": result["label"],
                        "color": shades[shade % len(shades)],
                    }
                )
        return results

    def _run_one(
        self, scenario: Scenario, key: str, custom: PIDGains
    ) -> dict[str, Any]:
        built = self.controller(key, custom)
        try:
            result = run_episode(built["env"], scenario, built["predictor"])
        finally:
            if built["close"]:
                built["env"].close()
        return {
            "key": key,
            "label": built["label"],
            "note": built["note"],
            "calibration": built["calibration"],
            "metrics": result.metrics,
            "trace": result.trace,
        }


def method_of(key: str) -> str:
    """Which of the three compared methods a controller key belongs to.

    Everything user-visible is grouped and coloured by this, so it is derived
    from the key in exactly one place rather than re-guessed per chart.
    """
    if key.startswith("pid_"):
        return "pid"
    if key.startswith("ppo_scheduler_"):
        return "scheduled"
    if key.startswith("ppo_direct_"):
        return "direct"
    raise ValueError(f"controller '{key}' belongs to no known method")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "path"


# --- figures ---------------------------------------------------------------


# The chart header is two stacked bands inside the top margin: the title on
# the first line, the horizontal legend on the second. Plotly positions each
# independently, so the two used to be given overlapping coordinates and the
# title was drawn straight through the legend entries. These constants keep the
# two apart in one place instead of at each of the nine call sites.
TITLE_BAND_PX = 26
LEGEND_BAND_PX = 30
CHART_TOP_PX = TITLE_BAND_PX + LEGEND_BAND_PX


def _base_figure(
    title: str, y_label: str, *, height: int, x_label: str, legend_rows: int = 1
) -> go.Figure:
    top = TITLE_BAND_PX + LEGEND_BAND_PX * legend_rows
    figure = go.Figure()
    figure.update_layout(
        template="plotly_dark",
        height=height + top - CHART_TOP_PX,
        margin=dict(l=55, r=15, t=top, b=40),
        paper_bgcolor=PANEL,
        plot_bgcolor="#0a0f1a",
        # Pinned to the top of the figure container rather than floated above
        # the plotting area, so it cannot drift down into the legend band when
        # the margin changes.
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=1.0,
            yanchor="top",
            yref="container",
            pad=dict(t=6),
            font=dict(size=13, color=MUTED),
        ),
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(title=x_label, gridcolor="#1e2a42", zeroline=False)
    figure.update_yaxes(title=y_label, gridcolor="#1e2a42", zeroline=False)
    return figure


def _show_legend(figure: go.Figure, *, x: float = 0.0) -> go.Figure:
    """Turn the legend on in the band the top margin already reserved for it."""
    figure.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=x,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            font=dict(size=10),
        ),
    )
    return figure


def _corridor_field(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marks = points[::CORRIDOR_STRIDE]
    padding = CORRIDOR_M * 1.35
    low, high = points.min(axis=0) - padding, points.max(axis=0) + padding
    grid_x = np.linspace(low[0], high[0], CORRIDOR_GRID)
    grid_y = np.linspace(low[1], high[1], CORRIDOR_GRID)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    distance_squared = np.full(mesh_x.shape, np.inf)
    for point_x, point_y in marks:
        np.minimum(
            distance_squared,
            (mesh_x - point_x) ** 2 + (mesh_y - point_y) ** 2,
            out=distance_squared,
        )
    return grid_x, grid_y, np.sqrt(distance_squared)


@lru_cache(maxsize=64)
def _corridor(spec_json: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _corridor_field(paths.build(json.loads(spec_json))["pts"])


def _event_bands(figure: go.Figure, scenario: Scenario, trace_end: float, label: bool) -> None:
    """Mark the disturbance events: a band for a single step, thin lines for a schedule."""
    counts: dict[str, int] = {}
    for event in scenario.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    for event in scenario.events:
        if float(event.start_s) > trace_end:
            continue
        tint = EVENT_TINTS.get(event.kind, WARN)
        if counts[event.kind] > 1:
            figure.add_vline(x=float(event.start_s), line=dict(color=tint, width=1, dash="dot"), opacity=0.45)
            continue
        end = event.end_s
        figure.add_vrect(
            x0=float(event.start_s),
            x1=min(float(end) if end is not None else trace_end, trace_end),
            fillcolor=tint,
            opacity=0.10,
            line_width=0,
            annotation_text=EVENT_LABELS.get(event.kind, event.kind) if label else None,
            annotation_position="top left",
            annotation_font=dict(size=10, color=tint),
        )


def trajectory_figure(
    results: list[dict[str, Any]], scenario: Scenario, *, height: int = 430
) -> go.Figure:
    spec_json = path_value(scenario.path)
    reference = paths.build(scenario.path)
    points = reference["pts"]
    figure = _base_figure(
        "trajectory", "y (m)", height=height, x_label="x (m)"
    )
    grid_x, grid_y, distance = _corridor(spec_json)
    figure.add_contour(
        x=grid_x,
        y=grid_y,
        z=distance,
        showscale=False,
        hoverinfo="skip",
        name="corridor",
        showlegend=False,
        line=dict(width=0),
        contours=dict(type="constraint", operation=">", value=CORRIDOR_M),
        fillcolor="rgba(139,151,173,0.13)",
    )
    figure.add_scatter(
        x=points[::10, 0],
        y=points[::10, 1],
        mode="lines",
        line=dict(color=PATH_COLOR, width=1.6, dash="dash"),
        hoverinfo="skip",
        name="reference path",
    )
    for result in results:
        figure.add_scatter(
            x=_values(result["trace"], "x"),
            y=_values(result["trace"], "y"),
            mode="lines",
            line=dict(color=result["color"], width=2.2),
            name=result["label"],
        )
    figure.add_scatter(
        x=[points[0, 0]], y=[points[0, 1]], mode="markers",
        marker=dict(color=GOOD, size=10), name="start", showlegend=False,
    )
    figure.add_scatter(
        x=[points[-1, 0]], y=[points[-1, 1]], mode="markers",
        marker=dict(color=WARN, size=10, symbol="square"), name="finish", showlegend=False,
    )
    figure.update_layout(hovermode="closest")
    _show_legend(figure)
    # Fitted to the path plus the corridor, so the plot is the path and not the
    # empty ground around it.
    margin = 1.2 * CORRIDOR_M
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    figure.update_xaxes(range=[float(low[0]), float(high[0])])
    figure.update_yaxes(range=[float(low[1]), float(high[1])], scaleanchor="x", scaleratio=1)
    return figure


def deviation_figure(rows: list[dict[str, Any]], scenario: Scenario) -> go.Figure:
    figure = _base_figure(
        "tracking error — e_ct (measured) vs true distance off path",
        "e (m)", height=285, x_label="time (s)",
    )
    times = _values(rows, "t")
    trace_end = times[-1] if times else 0.0
    _event_bands(figure, scenario, trace_end, label=True)
    for limit in (CORRIDOR_M, -CORRIDOR_M):
        figure.add_hline(y=limit, line=dict(color=BAD, width=1, dash="dot"))
    figure.add_hline(y=0.0, line=dict(color=LINE, width=1))
    figure.add_scatter(
        x=times, y=_values(rows, "distance_m"), mode="lines",
        line=dict(color=PATH_COLOR, width=1.2, dash="dot"), name="distance off path",
    )
    figure.add_scatter(
        x=times, y=_values(rows, "e_ct"), mode="lines",
        line=dict(color=ACCENT, width=1.8), name="e_ct (true)",
    )
    measured = _values(rows, "e_ct_measured")
    if measured:
        figure.add_scatter(
            x=times, y=measured, mode="lines",
            line=dict(color=WARN, width=1.0), opacity=0.65,
            name="e_ct (what the PID reads)",
        )
    _show_legend(figure)
    return figure


def speed_figure(rows: list[dict[str, Any]], scenario: Scenario) -> go.Figure:
    """Achieved speed against target, with the authority conflict shaded.

    Steering has first claim on the wheels, so the speed channel is what gives
    way in a corner. Shading the allocation-limited samples makes that visible
    instead of leaving it as an unexplained dip.
    """
    figure = _base_figure(
        "speed — achieved vs target", "v (m/s)", height=250, x_label="time (s)"
    )
    times = _values(rows, "t")
    trace_end = times[-1] if times else 0.0
    _event_bands(figure, scenario, trace_end, label=False)
    figure.add_hline(
        y=scenario.target_speed, line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text="target", annotation_position="top left",
        annotation_font=dict(size=10, color=MUTED),
    )
    spans = _allocation_spans(rows)
    for start, end in spans[:200]:
        figure.add_vrect(
            x0=start, x1=end, fillcolor=WARN, opacity=0.16, line_width=0, layer="below"
        )
    if spans:
        figure.add_annotation(
            text=f"shaded: steering has taken wheel authority ({len(spans)} spans)",
            showarrow=False, font=dict(size=10, color=WARN),
            xref="paper", yref="paper", x=0.99, y=1.13, xanchor="right",
        )
    figure.add_scatter(
        x=times, y=_values(rows, "speed_mps"), mode="lines",
        line=dict(color=GOOD, width=1.8), name="speed",
    )
    _show_legend(figure, x=0.18)
    return figure


def command_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """What was asked of the wheels and what they were actually given.

    Requested and applied are drawn together because the gap between them IS
    the saturation: a policy whose steering demand is clipped for half the
    corner is not the same as one that was never limited, and the two look
    identical on a trajectory plot.
    """
    figure = _base_figure(
        "wheel commands — requested vs applied", "u (normalised)",
        height=250, x_label="time (s)",
    )
    times = _values(rows, "t")
    limit = float(CONFIG.actuator_limit)
    for bound in (limit, -limit):
        figure.add_hline(y=bound, line=dict(color=BAD, width=1, dash="dot"))
    figure.add_hline(y=0.0, line=dict(color=LINE, width=1))
    for key, name, color, dash in (
        ("u_left", "left requested", ACCENT, "dot"),
        ("u_left_applied", "left applied", ACCENT, "solid"),
        ("u_right", "right requested", DELAY_TINT, "dot"),
        ("u_right_applied", "right applied", DELAY_TINT, "solid"),
    ):
        values = _values(rows, key)
        if not values:
            continue
        figure.add_scatter(
            x=times, y=values, mode="lines",
            line=dict(color=color, width=1.4, dash=dash), name=name,
        )
    _show_legend(figure)
    return figure


def gain_figure(rows: list[dict[str, Any]], calibration: GainCalibration) -> go.Figure:
    """What the controller did with its gains, against the box it may use.

    Flat lines here are a real result, not a missing plot: a scheduler that
    settles on one constant has learned nothing a fixed gain could not do, and
    a mean alone hides that perfectly.
    """
    figure = _base_figure(
        "steering gains — applied over the episode", "gain", height=250, x_label="time (s)"
    )
    times = _values(rows, "t")
    if not _values(rows, "kp"):
        figure.add_annotation(
            text="direct-mode controller: no PID gains to show",
            showarrow=False, font=dict(size=12, color=MUTED),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return figure
    for bound in (calibration.lower.kp, calibration.upper.kp):
        figure.add_hline(y=float(bound), line=dict(color=LINE, width=1, dash="dot"))
    for key, name, color in (("kp", "Kp", ACCENT), ("ki", "Ki", GOOD), ("kd", "Kd", WARN)):
        figure.add_scatter(
            x=times, y=_values(rows, key), mode="lines",
            line=dict(color=color, width=1.6), name=name,
        )
    _show_legend(figure, x=0.18)
    return figure


def disturbance_figure(rows: list[dict[str, Any]], scenario: Scenario) -> go.Figure:
    """When the plant changed under the controller.

    The two disturbances in this experiment are dead time and sensor noise, and
    both are step changes recorded per sample, so the trace itself is the
    timeline -- no need to trust the scenario declaration.
    """
    figure = _base_figure(
        "plant condition — dead time and sensor noise", "value",
        height=230, x_label="time (s)",
    )
    times = _values(rows, "t")
    trace_end = times[-1] if times else 0.0
    _event_bands(figure, scenario, trace_end, label=True)
    figure.add_scatter(
        x=times, y=[1000.0 * value for value in _values(rows, "actuator_delay_s")],
        mode="lines", line=dict(color=DELAY_TINT, width=1.8, shape="hv"),
        name="dead time (ms)",
    )
    figure.add_scatter(
        x=times, y=[1000.0 * value for value in _values(rows, "sensor_noise_m")],
        mode="lines", line=dict(color=NOISE_TINT, width=1.8, shape="hv"),
        name="sensor noise sigma (mm)",
    )
    _show_legend(figure)
    return figure


# --- method comparison -----------------------------------------------------
#
# The per-controller panels below answer "what did this controller do". These
# answer "how did the three methods differ", which is the question the thesis
# actually asks, and it cannot be read off three panels stacked vertically --
# a 4 mm difference in tracking error is invisible unless the traces share an
# axis. Every chart here is one signal, all selected controllers, one time base.


def _derive(rows: list[dict[str, Any]], key: str) -> list[float]:
    """One trace column, including the two that are differences of columns."""
    if key == "steer_differential":
        left = _values(rows, "u_left_applied")
        right = _values(rows, "u_right_applied")
        return [a - b for a, b in zip(left, right)]
    if key == "delay_ms":
        return [1000.0 * value for value in _values(rows, "actuator_delay_s")]
    if key == "noise_mm":
        return [1000.0 * value for value in _values(rows, "sensor_noise_m")]
    return _values(rows, key)


def comparison_figure(
    results: list[dict[str, Any]],
    scenario: Scenario,
    key: str,
    title: str,
    y_label: str,
    *,
    height: int = 270,
    scale: float = 1.0,
    label_events: bool = False,
    reference_lines: tuple[float, ...] = (),
    empty_note: str = "",
    log_y: bool = False,
    legend: bool = True,
) -> go.Figure:
    figure = _base_figure(title, y_label, height=height, x_label="time (s)")
    trace_end = max(
        (max(_values(result["trace"], "t"), default=0.0) for result in results),
        default=0.0,
    )
    _event_bands(figure, scenario, trace_end, label=label_events)
    for line in reference_lines:
        figure.add_hline(
            y=line,
            line=dict(color=BAD if line else LINE, width=1, dash="dot" if line else "solid"),
        )
    drawn = 0
    for result in results:
        values = _derive(result["trace"], key)
        if not values:
            continue
        drawn += 1
        figure.add_scatter(
            x=_values(result["trace"], "t"),
            y=[value * scale for value in values],
            mode="lines",
            line=dict(color=result["color"], width=1.8),
            name=result["label"],
        )
    if not drawn and empty_note:
        figure.add_annotation(
            text=empty_note,
            showarrow=False,
            font=dict(size=12, color=MUTED),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    if log_y and drawn:
        figure.update_yaxes(type="log")
    if legend:
        _show_legend(figure)
    else:
        # Colours are the same in every chart and named in the table and the
        # trajectory legend, so the band reserved for a legend is given back.
        figure.update_layout(margin=dict(t=TITLE_BAND_PX + 10), height=height - LEGEND_BAND_PX + 10)
    return figure


def disturbance_timeline(results: list[dict[str, Any]], scenario: Scenario) -> go.Figure:
    """Dead time (left axis) and sensor noise (right axis) over the run.

    The disturbance is the same for every controller, so one line each, taken
    from the longest trace.
    """
    rows = max((result["trace"] for result in results), key=len)
    figure = _base_figure("disturbance in force", "dead time (ms)", height=215, x_label="time (s)")
    times = _values(rows, "t")
    figure.add_scatter(
        x=times, y=_derive(rows, "delay_ms"), mode="lines",
        line=dict(color=DELAY_TINT, width=1.8, shape="hv"), name="dead time (ms)",
    )
    figure.add_scatter(
        x=times, y=_derive(rows, "noise_mm"), mode="lines", yaxis="y2",
        line=dict(color=NOISE_TINT, width=1.8, shape="hv"), name="sensor noise (mm)",
    )
    figure.update_layout(
        yaxis2=dict(title="noise (mm)", overlaying="y", side="right", showgrid=False, zeroline=False,
                    rangemode="tozero"),
        margin=dict(r=55),
    )
    figure.update_yaxes(rangemode="tozero")
    _show_legend(figure)
    return figure


GRAPH_CONFIG = {"displaylogo": False, "responsive": True}


def _graph(figure: go.Figure) -> dcc.Graph:
    return dcc.Graph(figure=figure, config=GRAPH_CONFIG)


def overview_charts(results: list[dict[str, Any]], scenario: Scenario) -> list[Any]:
    """What a run is judged by: where the robot went, how far off, under what, with which gain."""
    worst = max((max(_values(result["trace"], "distance_m"), default=0.0) for result in results), default=0.0)
    charts: list[Any] = [
        _graph(trajectory_figure(results, scenario, height=430)),
        _graph(comparison_figure(
            results, scenario, "distance_m", "tracking error — distance from the path", "distance (mm)",
            height=280, scale=1000.0, label_events=True, legend=False,
            # The corridor edge is drawn only when a run comes near it; otherwise
            # it would flatten every trace against the bottom of the chart.
            reference_lines=(1000.0 * CORRIDOR_M,) if worst > 0.5 * CORRIDOR_M else (),
        )),
    ]
    if scenario.events:
        charts.append(_graph(disturbance_timeline(results, scenario)))
    if any(result["method"] == "scheduled" for result in results):
        charts.append(_graph(comparison_figure(
            results, scenario, "kp", "proportional gain Kp (log scale)", "Kp",
            height=260, log_y=True, legend=False,
        )))
    return charts


def actuator_charts(results: list[dict[str, Any]], scenario: Scenario) -> list[Any]:
    limit = 2.0 * float(CONFIG.actuator_limit)
    return [
        _graph(comparison_figure(
            results, scenario, "omega_applied", "steering — yaw rate delivered to the robot",
            "omega (rad/s)", reference_lines=(0.0,),
        )),
        _graph(comparison_figure(
            results, scenario, "steer_differential",
            "wheel differential — left minus right (dotted: saturation)", "u_left - u_right",
            reference_lines=(limit, -limit, 0.0), legend=False,
        )),
    ]


def comparison_view(results: list[dict[str, Any]], scenario: Scenario) -> list[Any]:
    """Every overlaid chart, overview first."""
    return overview_charts(results, scenario) + actuator_charts(results, scenario)


# --- metric rendering ------------------------------------------------------


def _format_metric(key: str, value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool) or key == "finished":
        truthy = value if isinstance(value, bool) else str(value).lower() == "true"
        return "Yes" if truthy else "No"
    if isinstance(value, str) and not _is_number(value):
        return value
    number = float(value)
    if key.endswith("_pct"):
        return f"{number:.1f}%"
    if key.endswith("_m"):
        return f"{number * 100:.2f} cm"
    if key.endswith("_s"):
        return f"{number:.2f} s"
    if key.endswith("_fraction"):
        return f"{100.0 * number:.1f}%"
    if key.endswith("_mps"):
        return f"{number:.3f} m/s"
    if abs(number) >= 1000 or (number and abs(number) < 0.001):
        return f"{number:.3e}"
    return f"{number:.4f}"


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _metric(label: str, value: str) -> html.Div:
    return html.Div([html.B(label), html.Span(value)], className="metric")


def headline_metrics(metrics: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            _metric(label, _format_metric(key, metrics.get(key)))
            for key, label in HEADLINE_METRICS
        ],
        className="metrics",
    )


def full_metrics(metrics: dict[str, Any]) -> html.Details:
    shown = {key for key, _ in HEADLINE_METRICS}
    rest = [(key, value) for key, value in sorted(metrics.items()) if key not in shown]
    return html.Details(
        [
            html.Summary(f"All {len(metrics)} recorded metrics"),
            html.Div(
                [_metric(key.replace("_", " "), _format_metric(key, value)) for key, value in rest],
                className="metrics wide",
            ),
        ]
    )


SCOREBOARD_COLUMNS = (
    ("finished", "Completed", "{}"),
    ("failure_adjusted_error_m", "E_FA (mm)", "{:.2f}"),
    ("mean_distance_m", "Mean err (mm)", "{:.2f}"),
    ("max_distance_m", "Max err (mm)", "{:.2f}"),
    ("steer_total_variation_per_s", "Steer TV /s", "{:.2f}"),
    ("gain_total_variation_per_s", "Gain TV /s", "{:.2f}"),
)
# Columns whose stored unit is metres and whose readable unit is millimetres.
SCOREBOARD_MM = {"failure_adjusted_error_m", "mean_distance_m", "max_distance_m"}


def scoreboard(results: list[dict[str, Any]]) -> html.Table:
    """One row per controller, the numbers the comparison turns on.

    E_FA is the primary outcome and is reported in millimetres here because
    metres puts four leading zeros in front of every difference that matters.
    """
    header = html.Tr(
        [html.Th("Controller")]
        + [html.Th(label) for _key, label, _fmt in SCOREBOARD_COLUMNS]
        + [html.Th("3D view")]
    )
    body = []
    for result in results:
        cells = [
            html.Td(
                [
                    html.Span(
                        className="swatch",
                        style={"backgroundColor": result["color"]},
                    ),
                    result["label"],
                ],
                className="scoreboard-name",
            )
        ]
        for key, _label, fmt in SCOREBOARD_COLUMNS:
            value = result["metrics"].get(key)
            if key == "finished":
                text = "Yes" if value else "No"
            elif key.startswith("gain_") and result["method"] == "direct":
                # env.py records 0.0 here for direct mode. Printed as a number it
                # reads as "this policy held its gains perfectly steady", which is
                # the opposite of the truth: it has no gains at all.
                text = "n/a"
            elif value is None or value == "":
                text = "—"
            else:
                number = float(value)
                text = fmt.format(number * 1000.0 if key in SCOREBOARD_MM else number)
            cells.append(html.Td(text))
        cells.append(
            html.Td(
                html.Button(
                    "▶  MuJoCo",
                    id={"role": "replay", "key": result["key"], "at": "scoreboard"},
                    className="drawer-close replay-button",
                    n_clicks=0,
                )
                if result.get("key")
                else ""
            )
        )
        body.append(html.Tr(cells))
    return html.Table([html.Thead(header), html.Tbody(body)], className="scoreboard")


def _gain_activity(rows: list[dict[str, Any]]) -> str:
    parts = []
    for key, name in (("kp", "Kp"), ("ki", "Ki"), ("kd", "Kd")):
        values = _values(rows, key)
        if not values:
            continue
        low, high = min(values), max(values)
        parts.append(
            f"{name} {np.mean(values):.2f}"
            + (f" ({low:.2f}–{high:.2f})" if high - low > 1e-6 else " (constant)")
        )
    return "  ·  ".join(parts)


def controller_panel(
    result: dict[str, Any], scenario: Scenario, calibration: GainCalibration
) -> html.Section:
    rows = result["trace"]
    graph = {"displaylogo": False, "responsive": True}
    note = result["note"]
    activity = _gain_activity(rows)
    if activity:
        note += "  ·  " + activity
    return html.Section(
        [
            html.Div(
                [
                    html.H2(result["label"]),
                    # Per panel rather than in the drawer: the button is next to
                    # the charts for the controller it replays, so there is no
                    # step where you pick a controller twice and have to trust
                    # that the two picks agreed.
                    html.Button(
                        "▶  Watch in MuJoCo",
                        id={"role": "replay", "key": result["key"], "at": "panel"},
                        className="drawer-close replay-button",
                        n_clicks=0,
                    ),
                ],
                className="panel-head",
            ),
            html.P(note, className="controller-note"),
            headline_metrics(result["metrics"]),
            full_metrics(result["metrics"]),
            dcc.Graph(figure=deviation_figure(rows, scenario), config=graph),
            dcc.Graph(figure=speed_figure(rows, scenario), config=graph),
            dcc.Graph(figure=command_figure(rows), config=graph),
            *(
                [dcc.Graph(figure=gain_figure(rows, calibration), config=graph)]
                if _values(rows, "kp") else []
            ),
        ],
        className="comparison-panel",
    )


def scenario_summary(scenario: Scenario) -> list[str]:
    """The scenario in a few short phrases, for the line above the table."""
    delays = [event for event in scenario.events if event.kind == "delay_step"]
    noises = [event for event in scenario.events if event.kind == "noise_step"]
    parts = [path_label(scenario.path), f"{scenario.target_speed:.1f} m/s"]
    if scenario.condition == "nominal":
        return parts + ["no disturbance"]
    dynamic = len(delays) > 1 or len(noises) > 1
    if scenario.condition in {"delay", "combined"}:
        if dynamic:
            top = max([scenario.initial_delay_s] + [event.value for event in delays])
            parts.append(f"dead time redrawn every 1–3 s (up to {1000 * top:.0f} ms)")
        elif delays:
            parts.append(f"dead time {1000 * delays[0].value:g} ms from {delays[0].start_s:g} s")
        else:
            parts.append(f"dead time {1000 * scenario.initial_delay_s:g} ms")
    if scenario.condition in {"noise", "combined"}:
        if dynamic:
            top = max([scenario.initial_noise_std_m] + [event.value for event in noises])
            parts.append(f"noise redrawn every 1–3 s (up to {1000 * top:.2g} mm)")
        elif noises:
            parts.append(f"noise {1000 * noises[0].value:g} mm from {noises[0].start_s:g} s")
        else:
            parts.append(f"noise {1000 * scenario.initial_noise_std_m:g} mm")
    return parts


TAB_STYLE = {
    "padding": "9px 18px", "border": "none", "borderBottom": f"2px solid {LINE}",
    "background": "transparent", "color": MUTED, "fontSize": "0.9rem",
}
TAB_SELECTED = {**TAB_STYLE, "borderBottom": f"2px solid {ACCENT}", "color": "#e6edf7", "fontWeight": 600}


def interactive_view(
    results: list[dict[str, Any]], scenario: Scenario, runner: InteractiveRunner
) -> html.Div:
    """The table first, then three tabs: overview, actuators, one panel per controller."""
    def tab(label: str, children: list[Any], class_name: str = "tab-body") -> dcc.Tab:
        return dcc.Tab(
            html.Div(children, className=class_name), label=label,
            style=TAB_STYLE, selected_style=TAB_SELECTED,
        )

    return html.Div(
        [
            html.Section(
                [
                    html.Div(
                        [html.Span(part, className="chip") for part in scenario_summary(scenario)],
                        className="chips",
                    ),
                    scoreboard(results),
                ],
                className="comparison-panel overlay-panel",
            ),
            html.Section(
                dcc.Tabs(
                    [
                        tab("Overview", overview_charts(results, scenario)),
                        tab("Steering and wheels", actuator_charts(results, scenario)),
                        tab(
                            "Per controller",
                            [
                                # Each scheduler is drawn against its own gain range
                                # (narrow or wide), read from its checkpoint record.
                                controller_panel(result, scenario, result.get("calibration") or runner.calibration)
                                for result in results
                            ],
                            "tab-body comparison-grid",
                        ),
                    ],
                    className="result-tabs",
                ),
                className="comparison-panel overlay-panel",
            ),
        ]
    )


# --- controls --------------------------------------------------------------


def _declared_mark(minimum: float, maximum: float, default: float) -> dict[float, Any]:
    """A single labelled tick at the control's declared value.

    Returned empty when the default sits on an end of the track, where the tick
    would only restate the bound already printed there.
    """
    if not minimum < default < maximum:
        return {}
    style = {"color": MUTED, "fontSize": "10px", "whiteSpace": "nowrap"}
    return {default: {"label": f"{default:g}", "style": style}}


def slider_row(
    control_id: str, label: str, minimum: float, maximum: float, step: float, default: float
) -> html.Div:
    """A labelled slider. The slider carries its own number box for exact values."""
    return html.Div(
        [
            html.Div(label, className="slider-label"),
            dcc.Slider(
                minimum, maximum, step,
                value=default,
                id=f"s-{control_id}",
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="knob",
        id=f"row-{control_id}",
    )


def gain_controls(calibration: GainCalibration) -> tuple[tuple[Any, ...], ...]:
    base = calibration.robust.as_array()
    return tuple(
        (f"gain-{name}", GAIN_LABELS[index], CUSTOM_GAIN_BOUNDS[index][0], CUSTOM_GAIN_BOUNDS[index][1],
         GAIN_STEPS[index], float(base[index]))
        for index, name in enumerate(("kp", "ki", "kd"))
    )


SPLIT_LABELS = {"held_out": "test", "validation": "validation", "train": "training"}


def path_options() -> list[dict[str, str]]:
    """The paths of the thesis, test paths first, labelled with their split."""
    options: list[dict[str, str]] = []
    for split in ("held_out", "validation", "train"):
        for spec in PATH_SPLITS[split]:
            options.append(
                {"label": f"{SPLIT_LABELS[split]} · {path_label(spec)}", "value": path_value(dict(spec))}
            )
    return options


# --- app -------------------------------------------------------------------


def create_app(
    models_root: str | Path | None = None,
    calibration_path: str | Path | None = None,
    all_arms: bool = False,
) -> Dash:
    runner = InteractiveRunner(
        Path(models_root) if models_root else None, calibration_path, all_arms=all_arms
    )
    gains = gain_controls(runner.calibration)

    options = path_options()
    controller_options = runner.controller_options()
    default_selection = runner.default_selection()
    app = Dash(
        __name__, title="Learned vs. fixed PID",
        assets_folder=str(Path(__file__).resolve().parent / "dashboard_assets"),
    )
    controls = {control[0]: control for control in SCENARIO_CONTROLS}

    def select(label: str, control_id: str, value: Any, choices: list[dict[str, Any]]) -> html.Div:
        return html.Div(
            [
                html.Label(label, htmlFor=control_id),
                dcc.Dropdown(id=control_id, value=value, options=choices, clearable=False),
            ],
            className="selector",
        )

    app.layout = html.Div(
        [
            # Always on screen, above the drawer: one button opens and closes
            # the controls wherever the page or the drawer is scrolled to.
            html.Button("☰  Controls", id="panel-toggle", className="panel-toggle", n_clicks=0),
            dcc.Store(id="layout-tick"),
            html.Header(
                [
                    html.H1("Path following under delay and noise"),
                    html.P(
                        "Fixed PID, PPO gain scheduler and direct PPO on the same path "
                        "and the same disturbance.",
                        className="kicker",
                    ),
                ],
                className="masthead",
            ),
            html.Aside(
                [
                    # Keeps the top of the drawer clear for the Controls button.
                    html.Div(className="drawer-head"),
                    html.Div("Scenario", className="section-title"),
                    select("Path", "path-select", options[0]["value"], options),
                    select(
                        "Speed", "speed-select", CONFIG.target_speeds[1],
                        [{"label": f"{speed:.1f} m/s", "value": speed} for speed in CONFIG.target_speeds],
                    ),
                    select(
                        "Disturbance", "condition-select", "combined",
                        [
                            {"label": "none", "value": "nominal"},
                            {"label": "delay", "value": "delay"},
                            {"label": "noise", "value": "noise"},
                            {"label": "delay and noise", "value": "combined"},
                        ],
                    ),
                    select(
                        "Timing", "mode-select", "stationary",
                        [
                            {"label": "stationary — whole run", "value": "stationary"},
                            {"label": "transient — steps in", "value": "transient"},
                            {"label": "dynamic — redrawn every 1–3 s", "value": "dynamic"},
                        ],
                    ),
                    slider_row(*controls["delay"]),
                    slider_row(*controls["noise"]),
                    slider_row(*controls["event-time"]),
                    html.Div("Controllers", className="section-title"),
                    *[
                        html.Div(
                            [
                                html.Label(METHOD_LABELS[method]),
                                # Always in the tree, even with no options: Dash
                                # raises on a callback whose State id is absent.
                                dcc.Dropdown(
                                    id=f"select-{method}",
                                    options=controller_options[method],
                                    value=default_selection[method],
                                    multi=True,
                                    placeholder=(
                                        "not compared"
                                        if controller_options[method]
                                        else "no trained checkpoint found"
                                    ),
                                    disabled=not controller_options[method],
                                ),
                            ],
                            className="selector method-group",
                        )
                        for method in METHOD_ORDER
                    ],
                    html.Div(runner.model_error, className="group-hint") if runner.model_error else html.Div(),
                    html.Details(
                        [
                            html.Summary("Advanced"),
                            slider_row(*controls["seed"]),
                            html.Div("Gains of the hand-tuned PID", className="group-hint"),
                            *[slider_row(*control) for control in gains],
                            html.Button(
                                "Reset to PID-R", id="gain-reset",
                                className="drawer-close", n_clicks=0,
                            ),
                        ],
                        className="advanced",
                    ),
                    html.Div(
                        html.Button("▶  Run", id="run-button", className="run-button", n_clicks=0),
                        className="run-bar",
                    ),
                ],
                id="controls-drawer",
                className="controls panel drawer open",
            ),
            html.Main(
                [
                    html.Div(id="replay-status", className="replay-status"),
                    dcc.Loading(html.Div(id="tab-content"), type="dot", color=ACCENT),
                ],
                className="results",
            ),
        ],
        id="app-root",
        className="app drawer-open",
    )

    @app.callback(
        Output("controls-drawer", "className"),
        Output("app-root", "className"),
        Input("panel-toggle", "n_clicks"),
        State("controls-drawer", "className"),
        prevent_initial_call=True,
    )
    def toggle_drawer(_clicks: int, current: str):
        """The Controls button opens and closes the drawer; the page makes room for it."""
        if "open" in (current or "").split():
            return "controls panel drawer", "app"
        return "controls panel drawer open", "app drawer-open"

    # The charts size themselves on a window resize only, so one is announced
    # once the drawer has finished sliding.
    app.clientside_callback(
        "function(name) { setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 280);"
        " return name; }",
        Output("layout-tick", "data"),
        Input("app-root", "className"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("row-delay", "style"),
        Output("row-noise", "style"),
        Output("row-event-time", "style"),
        Input("mode-select", "value"),
        Input("condition-select", "value"),
    )
    def show_relevant_sliders(evaluation_mode: str, condition: str):
        """Only the sliders the chosen disturbance uses are shown."""
        hidden: dict[str, str] = {"display": "none"}
        return (
            {} if condition in {"delay", "combined"} else hidden,
            {} if condition in {"noise", "combined"} else hidden,
            {} if evaluation_mode == "transient" and condition != "nominal" else hidden,
        )

    for control_id, _label, _minimum, _maximum, _step, default in gains:
        app.callback(
            Output(f"s-{control_id}", "value"),
            Input("gain-reset", "n_clicks"),
            prevent_initial_call=True,
        )(lambda _clicks, base=default: base)

    @app.callback(
        Output("tab-content", "children"),
        Input("run-button", "n_clicks"),
        State("path-select", "value"),
        State("speed-select", "value"),
        State("mode-select", "value"),
        State("condition-select", "value"),
        State("s-delay", "value"),
        State("s-noise", "value"),
        State("s-event-time", "value"),
        State("s-seed", "value"),
        State("select-pid", "value"),
        State("select-scheduled", "value"),
        State("select-direct", "value"),
        State("s-gain-kp", "value"),
        State("s-gain-ki", "value"),
        State("s-gain-kd", "value"),
    )
    def render(
        run_clicks,
        path_spec, speed, evaluation_mode, condition,
        delay_ms, noise_mm, event_time, seed,
        pid_keys, scheduled_keys, direct_keys, kp, ki, kd,
    ):
        controller_keys = (pid_keys or []) + (scheduled_keys or []) + (direct_keys or [])
        if not run_clicks:
            return html.Div(
                "Press Run to compare the selected controllers. The first run takes "
                "a few seconds while the checkpoints are loaded.",
                className="message",
            )
        try:
            scenario = runner.scenario_from_controls(
                path_spec, speed, condition, evaluation_mode,
                delay_ms, noise_mm, event_time, seed,
            )
            results_list = runner.run(
                scenario, controller_keys or [], (float(kp), float(ki), float(kd))
            )
            return interactive_view(results_list, scenario, runner)
        except (ValueError, KeyError, FileNotFoundError) as error:
            return html.Div(str(error), className="message error")

    @app.callback(
        Output("replay-status", "children"),
        Input({"role": "replay", "key": ALL, "at": ALL}, "n_clicks"),
        State("path-select", "value"),
        State("speed-select", "value"),
        State("mode-select", "value"),
        State("condition-select", "value"),
        State("s-delay", "value"),
        State("s-noise", "value"),
        State("s-event-time", "value"),
        State("s-seed", "value"),
        State("s-gain-kp", "value"),
        State("s-gain-ki", "value"),
        State("s-gain-kd", "value"),
        prevent_initial_call=True,
    )
    def launch_replay(
        clicks, path_spec, speed, evaluation_mode, condition,
        delay_ms, noise_mm, event_time, seed, kp, ki, kd,
    ):
        """Open the MuJoCo viewer on the episode this panel is showing.

        A separate process because MuJoCo's viewer wants the main thread and
        would fight the Dash server for it. The scenario is passed as the same
        drawer values the charts were built from, and `replay.py` rebuilds it
        through `scenario_from_controls`, so the window cannot end up showing a
        different episode from the one on screen.
        """
        triggered = ctx.triggered_id
        # The ALL pattern fires once on render with every count at zero. Without
        # this a viewer window would open by itself every time a run finished.
        # `clicks` is normally the list of counts, one per matched button, but
        # is a bare int when a single component matches the pattern.
        counts = clicks if isinstance(clicks, (list, tuple)) else [clicks]
        if not isinstance(triggered, dict) or not any(counts or []):
            return no_update
        command = [
            sys.executable,
            "-m", "pathfollow.replay",
            "--controller", str(triggered["key"]),
            "--path", str(path_spec),
            "--speed", str(float(speed)),
            "--mode", str(evaluation_mode),
            "--condition", str(condition),
            "--delay-ms", str(float(delay_ms)),
            "--noise-mm", str(float(noise_mm)),
            "--event-time", str(float(event_time)),
            "--noise-seed", str(float(seed)),
            "--gains", f"{float(kp)},{float(ki)},{float(kd)}",
            "--models-root", str(runner.models_root),
        ] + (["--all-arms"] if runner.all_arms else [])
        try:
            subprocess.Popen(command, cwd=str(config.ROOT))
        except OSError as error:
            return html.Div(f"Could not start the viewer: {error}", className="message error")
        return html.Div(
            "Opening the MuJoCo window, give it a few seconds. The pink ghost is "
            "where the noisy measurement puts the robot.",
            className="message",
        )

    @app.server.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        default=str(DEFAULT_MODELS_ROOT),
        help="Directory scanned for <arm>/seed_<n>/best_model.zip checkpoints.",
    )
    parser.add_argument(
        "--all-arms", action="store_true",
        help="Also offer the ablation controllers (history, gain range, budget, point training).",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help=f"Gain file of PID-N and PID-R (default: {DEFAULT_CALIBRATION}).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.models_root, args.calibration, args.all_arms)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
