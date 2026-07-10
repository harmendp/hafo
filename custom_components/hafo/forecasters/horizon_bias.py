"""Horizon bias correction forecaster.

This forecaster does not build a forecast from scratch. Instead it wraps a
set of existing forecast entities (typically the "today" / "tomorrow" /
"day after tomorrow" entities open-meteo-based solar forecast integrations
provide for a single roof orientation) and corrects them for systematic
local shading — e.g. buildings blocking the sun during specific hours — that
a generic weather-based forecast cannot know about.

One config entry corresponds to ONE roof orientation (e.g. "tuinhuis oost").
If you have multiple orientations with separate actual-production
measurements, create one entry per orientation and sum the resulting
corrected forecast sensors downstream (e.g. in HAEO's PV source list).

Why this can't just reuse recorder statistics (like HistoricalAveragedForecaster
does for its source entity):
- Recorder statistics track an entity's *state* over time, not its
  attributes. Day-based solar forecast entities typically expose the
  interesting data (a sub-daily forecast curve) via the `forecast`
  attribute, while the state itself is something like a running daily
  total — not a value comparable to a specific 15-min slot.
- So there is no way to look up "what did this entity predict for
  10:15 last Tuesday" from the recorder after the fact; once a day rolls
  over, the "today" entity's forecast attribute is overwritten with new
  data and the old prediction is gone.
- This forecaster therefore keeps its own small persistent archive
  (`homeassistant.helpers.storage.Store`): every update cycle it snapshots
  the currently visible forecast points, and once a point's time has
  passed, freezes its last-seen predicted value and later fills in the
  matching actual production once recorder statistics for that slot become
  available. That archive is the training data for the bias table.

Bias table:
- Ratio actual/forecast, one value per (slot, day), first averaged within
  the day (so 3-4 correlated 5-min samples from the same day/slot count as
  ONE independent observation), then recency-weighted and MAD-outlier
  filtered across days.
- Grouped by quarter-of-day (0-95, time of day only, NOT weekday): shading
  from buildings depends on sun elevation/azimuth, not on which weekday it
  is. With RECENCY_DECAY = 0.85 a sample is already down to ~1% weight
  after 28 days, so seasonal drift within the learning window is small
  enough to ignore.
- Clamped to [0.2, 1.0]: shading can only reduce expected PV output
  relative to the weather-only forecast, never increase it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.components.recorder.statistics import StatisticsRow, statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.recorder import get_instance
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from custom_components.hafo.const import (
    ATTR_FORECAST,
    CONF_FORECAST_ENTITIES,
    CONF_HISTORY_DAYS,
    CONF_MIN_DAYS_PER_BUCKET,
    CONF_REFERENCE_ENTITY,
    DEFAULT_BIAS_HISTORY_DAYS,
    DEFAULT_MIN_DAYS_PER_BUCKET,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

StatisticsLike = Mapping[str, Any]

QUARTERS_PER_DAY = 96
RECENCY_DECAY = 0.85
MIN_DAYS_PER_BUCKET_DEFAULT = DEFAULT_MIN_DAYS_PER_BUCKET
BIAS_MIN_FACTOR = 0.2
BIAS_MAX_FACTOR = 1.0
MIN_FORECAST_FOR_RATIO = 5.0

# 5-min statistics for a slot are normally available within a few minutes;
# 20 min is a safe margin before we try to look up the matching actual value.
ACTUAL_LOOKUP_DELAY = timedelta(minutes=20)

# Archive entries older than this many days (beyond history_days) are
# dropped to keep the persisted store from growing unbounded.
ARCHIVE_RETENTION_MARGIN_DAYS = 2

STORAGE_VERSION = 1


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    """A single point in a forecast time series."""

    time: datetime
    value: float


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Result of a forecast operation."""

    forecast: list[ForecastPoint]
    source_entity: str
    history_days: int
    generated_at: datetime


def get_forecast_points(hass: HomeAssistant, entity_id: str) -> list[ForecastPoint]:
    """Read the current `forecast` attribute points from a forecast entity."""
    state = hass.states.get(entity_id)
    if state is None:
        return []

    raw_points = state.attributes.get(ATTR_FORECAST, [])
    points: list[ForecastPoint] = []
    for raw in raw_points:
        try:
            time = dt_util.parse_datetime(raw["time"])
            value = float(raw["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if time is not None:
            points.append(ForecastPoint(time=time, value=value))
    return points


def get_combined_forecast_points(hass: HomeAssistant, entity_ids: Sequence[str]) -> list[ForecastPoint]:
    """Concatenate forecast points from multiple day-based entities.

    Typical use: "today" / "tomorrow" / "day after tomorrow" entities for a
    single orientation, which don't overlap in time. If two entities do
    report a point for the exact same timestamp, the later entity in
    `entity_ids` wins (last write).
    """
    by_time: dict[datetime, float] = {}
    for entity_id in entity_ids:
        for point in get_forecast_points(hass, entity_id):
            by_time[point.time] = point.value
    return [ForecastPoint(time=t, value=v) for t, v in sorted(by_time.items())]


def _quarter_of_day(dt: datetime) -> int:
    """Return the 15-min slot index (0-95) for a datetime's time of day."""
    return dt.hour * 4 + dt.minute // 15


def _remove_outliers(values: list[float], threshold: float = 2.0) -> list[float]:
    """Verwijder uitschieters op basis van de Median Absolute Deviation (MAD).

    Zelfde aanpak als in HistoricalAveragedForecaster: robuuster dan
    standaarddeviatie, minimaal 4 samples nodig anders niets weggooien.
    """
    if len(values) < 4:
        return values
    sorted_vals = sorted(values)
    median = sorted_vals[len(sorted_vals) // 2]
    deviations = sorted([abs(v - median) for v in values])
    mad = deviations[len(deviations) // 2]
    if mad == 0:
        return values
    return [v for v in values if abs(v - median) / mad <= threshold]


def _parse_stat_time(start: Any) -> datetime | None:
    """Parse a statistics start value to datetime."""
    if isinstance(start, datetime):
        return start
    try:
        return datetime.fromtimestamp(float(start), tz=dt_util.get_default_time_zone())
    except (TypeError, ValueError):
        return None


async def _get_actual_stats(
    hass: HomeAssistant,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[datetime, float]:
    """Fetch 5-minute statistics for an entity, indexed by start time."""
    recorder = get_instance(hass)
    statistics: dict[str, list[StatisticsRow]] = await recorder.async_add_executor_job(
        lambda: statistics_during_period(
            hass,
            start_time,
            end_time,
            {entity_id},
            "5minute",
            None,
            {"mean"},
        )
    )
    result: dict[datetime, float] = {}
    for stat in statistics.get(entity_id, []):
        dt_start = _parse_stat_time(stat.get("start"))
        mean = stat.get("mean")
        if dt_start is not None and mean is not None:
            result[dt_start] = float(mean)
    return result


def _actual_average_for_slot(actual_by_time: Mapping[datetime, float], slot_start: datetime) -> float | None:
    """Average the three 5-min actual buckets covering a 15-min forecast slot."""
    values = [
        actual_by_time[t]
        for t in (slot_start, slot_start + timedelta(minutes=5), slot_start + timedelta(minutes=10))
        if t in actual_by_time
    ]
    return sum(values) / len(values) if values else None


def build_bias_table(
    archive: Mapping[str, dict[str, float | None]],
    now: datetime,
    min_days_per_bucket: int = MIN_DAYS_PER_BUCKET_DEFAULT,
) -> dict[int, float]:
    """Build a recency-weighted actual/forecast ratio per quarter-of-day slot.

    Args:
        archive: Mapping of ISO timestamp -> {"forecast": float, "actual": float|None}.
            Only entries with a non-None "actual" contribute.
        now: Current time, used for recency weighting.
        min_days_per_bucket: Minimum number of distinct days required in a
            slot before its correction factor is applied.

    Returns:
        Mapping of quarter-of-day slot (0-95) to correction factor, clamped
        to [BIAS_MIN_FACTOR, BIAS_MAX_FACTOR]. Slots without enough distinct
        days of data are omitted (caller should treat missing = 1.0).

    """
    # Stap 1: per (slot, dag) de ratio's verzamelen en middelen tot 1
    # dag-gemiddelde. Meerdere records op dezelfde dag/slot zijn geen
    # onafhankelijke waarnemingen van de schaduw-ratio.
    raw_by_slot_day: dict[tuple[int, Any], list[float]] = {}
    for time_key, record in archive.items():
        actual = record.get("actual")
        forecast = record.get("forecast")
        if actual is None or forecast is None or forecast < MIN_FORECAST_FOR_RATIO:
            continue
        point_time = dt_util.parse_datetime(time_key)
        if point_time is None:
            continue
        ratio = actual / forecast
        slot = _quarter_of_day(point_time)
        key = (slot, point_time.date())
        raw_by_slot_day.setdefault(key, []).append(ratio)

    daily_samples: dict[int, list[tuple[float, float]]] = {}
    for (slot, day), ratios in raw_by_slot_day.items():
        daily_ratio = sum(ratios) / len(ratios)
        age_days = (now.date() - day).days
        weight = RECENCY_DECAY**age_days
        daily_samples.setdefault(slot, []).append((daily_ratio, weight))

    bias_table: dict[int, float] = {}
    for slot, samples in daily_samples.items():
        if len(samples) < min_days_per_bucket:
            continue
        ratios = [r for r, _ in samples]
        clean_ratios = set(_remove_outliers(ratios))
        filtered = [(r, w) for r, w in samples if r in clean_ratios]
        if not filtered:
            continue
        weighted_avg = sum(r * w for r, w in filtered) / sum(w for _, w in filtered)
        bias_table[slot] = min(BIAS_MAX_FACTOR, max(BIAS_MIN_FACTOR, weighted_avg))

    return bias_table


def apply_bias(
    points: Sequence[ForecastPoint],
    bias_table: Mapping[int, float],
    now: datetime,
) -> list[ForecastPoint]:
    """Apply the learned bias table to future forecast points.

    Points in the past are passed through unchanged. Missing bucket → factor
    1.0 (no correction, e.g. too little history yet).
    """
    corrected: list[ForecastPoint] = []
    for point in points:
        if point.time < now:
            corrected.append(point)
            continue
        factor = bias_table.get(_quarter_of_day(point.time), 1.0)
        corrected.append(ForecastPoint(time=point.time, value=point.value * factor))
    return corrected


class HorizonBiasForecaster(DataUpdateCoordinator[ForecastResult | None]):
    """Forecaster that corrects a set of forecast entities for local shading.

    Covers ONE roof orientation per config entry (see module docstring for
    why: the archive is keyed against a single reference_entity). Combines
    the configured forecast_entities (typically today/tomorrow/day-after
    for that orientation) into one continuous series, archives their
    predicted values before they'd otherwise be overwritten, matches them
    against actual production once available, and applies the learned
    per-time-of-day correction to future points.

    Update interval: 15 minutes.
    """

    UPDATE_INTERVAL = timedelta(minutes=15)

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the forecaster."""
        self._entry = entry
        self._forecast_entities: list[str] = list(entry.data[CONF_FORECAST_ENTITIES])
        self._reference_entity: str = entry.data[CONF_REFERENCE_ENTITY]
        self._history_days: int = int(
            entry.options.get(CONF_HISTORY_DAYS, entry.data.get(CONF_HISTORY_DAYS, DEFAULT_BIAS_HISTORY_DAYS))
        )
        self._min_days_per_bucket: int = int(
            entry.options.get(
                CONF_MIN_DAYS_PER_BUCKET,
                entry.data.get(CONF_MIN_DAYS_PER_BUCKET, DEFAULT_MIN_DAYS_PER_BUCKET),
            )
        )
        self._store: Store[dict[str, dict[str, float | None]]] = Store(
            hass,
            STORAGE_VERSION,
            f"{DOMAIN}_{entry.entry_id}_horizon_bias_archive",
        )
        self._archive: dict[str, dict[str, float | None]] = {}
        self._archive_loaded = False

        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{entry.entry_id}_horizon_bias",
            update_interval=self.UPDATE_INTERVAL,
            config_entry=entry,
        )

    @property
    def source_entity(self) -> str:
        """Return the first forecast entity (used by sensor.py for unit/device_class)."""
        return self._forecast_entities[0]

    @property
    def forecast_entities(self) -> list[str]:
        """Return all configured forecast entities for this orientation."""
        return self._forecast_entities

    @property
    def history_days(self) -> int:
        """Return the number of history days used to learn the bias table."""
        return self._history_days

    @property
    def min_days_per_bucket(self) -> int:
        """Return the minimum distinct days required per quarter-of-day bucket."""
        return self._min_days_per_bucket

    @property
    def entry(self) -> ConfigEntry:
        """Return the config entry."""
        return self._entry

    async def _async_update_data(self) -> ForecastResult:
        """Fetch and update forecast data."""
        if not self._archive_loaded:
            self._archive = await self._store.async_load() or {}
            self._archive_loaded = True

        result = await self._generate_forecast()
        _LOGGER.debug(
            "Generated horizon-bias-corrected forecast for %s with %d points",
            self._forecast_entities[0],
            len(result.forecast),
        )
        return result

    def _archive_current_points(self, combined_points: Sequence[ForecastPoint], now: datetime) -> None:
        """Snapshot currently visible forecast points into the archive.

        Future points keep getting refreshed (closer to real-time = better
        estimate). Once a point's time has passed, its forecast value is
        frozen (first time we see it as "past" without an existing entry).
        """
        for point in combined_points:
            key = point.time.isoformat()
            if point.time > now:
                existing = self._archive.get(key, {})
                self._archive[key] = {"forecast": point.value, "actual": existing.get("actual")}
            elif key not in self._archive:
                self._archive[key] = {"forecast": point.value, "actual": None}

    async def _backfill_actuals(self, now: datetime) -> None:
        """Fill in actual production for archive entries whose time has passed."""
        pending_times = [
            dt_util.parse_datetime(key)
            for key, record in self._archive.items()
            if record.get("actual") is None
        ]
        pending_times = [t for t in pending_times if t is not None and now - t >= ACTUAL_LOOKUP_DELAY]
        if not pending_times:
            return

        window_start = min(pending_times)
        window_end = max(pending_times) + timedelta(minutes=15)
        actual_by_time = await _get_actual_stats(self.hass, self._reference_entity, window_start, window_end)

        for t in pending_times:
            avg = _actual_average_for_slot(actual_by_time, t)
            if avg is not None:
                self._archive[t.isoformat()]["actual"] = avg

    def _prune_archive(self, now: datetime) -> None:
        """Drop archive entries older than the configured history window."""
        cutoff = now - timedelta(days=self._history_days + ARCHIVE_RETENTION_MARGIN_DAYS)
        self._archive = {
            key: record
            for key, record in self._archive.items()
            if (parsed := dt_util.parse_datetime(key)) is not None and parsed >= cutoff
        }

    async def _generate_forecast(self) -> ForecastResult:
        """Generate a bias-corrected forecast."""
        now = dt_util.now()

        combined_points = get_combined_forecast_points(self.hass, self._forecast_entities)
        if not combined_points:
            msg = f"No forecast points available on {self._forecast_entities}"
            raise ValueError(msg)

        self._archive_current_points(combined_points, now)
        await self._backfill_actuals(now)
        self._prune_archive(now)
        await self._store.async_save(self._archive)

        bias_table = build_bias_table(self._archive, now, self._min_days_per_bucket)

        future_points = [p for p in combined_points if p.time >= now]
        corrected = apply_bias(future_points, bias_table, now)

        _LOGGER.debug(
            "Bias table for %s covers %d/%d quarter-of-day slots (archive size: %d)",
            self._forecast_entities[0],
            len(bias_table),
            QUARTERS_PER_DAY,
            len(self._archive),
        )

        return ForecastResult(
            forecast=corrected,
            source_entity=self._forecast_entities[0],
            history_days=self._history_days,
            generated_at=now,
        )

    def cleanup(self) -> None:
        """Clean up coordinator resources."""
        _LOGGER.debug("Cleaning up horizon-bias forecaster for %s", self._forecast_entities)
