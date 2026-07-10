"""Horizon bias correction forecaster.

This forecaster does not build a forecast from scratch. Instead it wraps an
existing forecast entity (typically an open-meteo based PV forecast) and
corrects it for systematic local shading — e.g. buildings blocking the sun
during specific hours — that a generic weather-based forecast cannot know
about.

Approach:
- Compare historical actual production (reference_entity) against the
  historical forecast (forecast_entity, read from recorder statistics of its
  own state) for the same 5-minute slots.
- Group the ratio actual/forecast by quarter-of-day (0-95, i.e. time of day
  only, NOT weekday). Shading from buildings depends on sun elevation/azimuth
  (time of day + slowly-shifting time of year), not on which weekday it is.
- Recency-weighted averaging + MAD outlier filtering, same as
  HistoricalAveragedForecaster, so cloudy/overcast days don't skew the bias.
- No separate week-of-year dimension: with RECENCY_DECAY = 0.85 a sample is
  already down to ~1% weight after 28 days, so seasonal drift within the
  learning window is small enough to ignore. Revisit if BIAS_HISTORY_DAYS is
  ever increased significantly.
- The learned per-slot factor is applied to the *future* points of the
  forecast entity's own `forecast` attribute, clamped to [0.2, 1.0]: shading
  can only reduce expected PV output relative to the weather-only forecast,
  never increase it, so the factor should never push the value up.
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
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from custom_components.hafo.const import (
    ATTR_FORECAST,
    CONF_FORECAST_ENTITY,
    CONF_HISTORY_DAYS,
    CONF_MIN_DAYS_PER_BUCKET,
    CONF_REFERENCE_ENTITY,
    DEFAULT_BIAS_HISTORY_DAYS,
    DEFAULT_MIN_DAYS_PER_BUCKET,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

StatisticsLike = Mapping[str, Any]

# Aantal 15-min slots per dag (0-95). Bucketing is uitsluitend op tijdstip-
# van-de-dag, niet op weekdag: schaduw door gebouwen hangt af van zonshoek,
# niet van welke dag van de week het is.
QUARTERS_PER_DAY = 96

# Zelfde decay als HistoricalAveragedForecaster: recente dagen wegen zwaarder.
RECENCY_DECAY = 0.85

# Standaardwaarde, instelbaar per config entry via CONF_MIN_DAYS_PER_BUCKET
# (zie const.py). Onder de 4-5 dagen kan de MAD-outlierfilter sowieso niet
# goed werken (die heeft zelf minimaal 4 samples nodig), dus een lagere
# waarde dan het HA-default van DEFAULT_MIN_DAYS_PER_BUCKET wordt afgeraden.
MIN_DAYS_PER_BUCKET_DEFAULT = DEFAULT_MIN_DAYS_PER_BUCKET

# Correctiefactor mag nooit boven 1.0 komen (schaduw verlaagt opbrengst,
# verhoogt hem nooit) en niet onder 0.2 zakken (voorkomt dat één extreme
# uitschieter een slot permanent naar (bijna) nul corrigeert).
BIAS_MIN_FACTOR = 0.2
BIAS_MAX_FACTOR = 1.0

# Forecast-waarden onder deze drempel (W) worden genegeerd bij het opbouwen
# van de bias-tabel: ratio's rond zonsopgang/-ondergang zijn numeriek
# instabiel (deling door bijna-nul) en dragen niets bij.
MIN_FORECAST_FOR_RATIO = 5.0


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


async def _get_statistics(
    hass: HomeAssistant,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
) -> list[StatisticsLike]:
    """Fetch 5-minute statistics for an entity."""
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
    return list(statistics.get(entity_id, []))


def _parse_stat_time(start: Any) -> datetime | None:
    """Parse a statistics start value to datetime."""
    if isinstance(start, datetime):
        return start
    try:
        return datetime.fromtimestamp(float(start), tz=dt_util.get_default_time_zone())
    except (TypeError, ValueError):
        return None


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


def _quarter_of_day(dt: datetime) -> int:
    """Return the 15-min slot index (0-95) for a datetime's time of day."""
    return dt.hour * 4 + dt.minute // 15


def build_bias_table(
    actual_stats: Sequence[StatisticsLike],
    forecast_stats: Sequence[StatisticsLike],
    now: datetime,
    min_days_per_bucket: int = MIN_DAYS_PER_BUCKET_DEFAULT,
) -> dict[int, float]:
    """Build a recency-weighted actual/forecast ratio per quarter-of-day slot.

    Args:
        actual_stats: 5-min statistics of the ground-truth production sensor.
        forecast_stats: 5-min statistics of the forecast entity's own state
            history (i.e. what it was predicting at that point in time).
        now: Current time, used for recency weighting.
        min_days_per_bucket: Minimum number of distinct days required in a
            slot before its correction factor is applied.

    Returns:
        Mapping of quarter-of-day slot (0-95) to correction factor, already
        clamped to [BIAS_MIN_FACTOR, BIAS_MAX_FACTOR]. Slots without enough
        distinct days of data are omitted (caller should treat missing = 1.0).

    """
    # Forecast-stats indexeren op exacte starttijd zodat we per 5-min slot
    # kunnen matchen met de actual-stats (recorder gebruikt dezelfde
    # afgeronde tijdstippen voor beide entiteiten).
    forecast_by_time: dict[datetime, float] = {}
    for stat in forecast_stats:
        dt_start = _parse_stat_time(stat.get("start"))
        mean = stat.get("mean")
        if dt_start is not None and mean is not None:
            forecast_by_time[dt_start] = float(mean)

    # Stap 1: ruwe 5-min ratio's verzamelen per (slot, dag). Dit is nog GEEN
    # onafhankelijke steekproef — de 3 samples binnen één kwartier op één dag
    # horen bij hetzelfde moment van hetzelfde weer.
    raw_by_slot_day: dict[tuple[int, Any], list[float]] = {}
    for stat in actual_stats:
        dt_start = _parse_stat_time(stat.get("start"))
        actual_mean = stat.get("mean")
        if dt_start is None or actual_mean is None:
            continue

        forecast_mean = forecast_by_time.get(dt_start)
        if forecast_mean is None or forecast_mean < MIN_FORECAST_FOR_RATIO:
            continue

        ratio = float(actual_mean) / forecast_mean
        slot = _quarter_of_day(dt_start)
        key = (slot, dt_start.date())
        raw_by_slot_day.setdefault(key, []).append(ratio)

    # Stap 2: middelen tot 1 ratio per (slot, dag) — dát is de onafhankelijke
    # waarneming die we willen tellen en wegen.
    daily_samples: dict[int, list[tuple[float, float]]] = {}
    for (slot, day), ratios in raw_by_slot_day.items():
        daily_ratio = sum(ratios) / len(ratios)
        # Leeftijd in dagen t.o.v. vandaag, voor de recency-weging.
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


def apply_bias(
    points: Sequence[ForecastPoint],
    bias_table: Mapping[int, float],
    now: datetime,
) -> list[ForecastPoint]:
    """Apply the learned bias table to future forecast points.

    Points in the past are passed through unchanged (nothing to correct).
    Missing bucket → factor 1.0 (no correction, e.g. too little history yet).
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
    """Forecaster that corrects another forecast entity for local shading.

    Learns a per-time-of-day correction factor from the historical ratio of
    actual production to forecast, and applies it to the wrapped forecast
    entity's future points.

    Update interval: 15 minutes (matches HistoricalAveragedForecaster; the
    bias table itself changes slowly, but the underlying forecast entity may
    update more often and we want to pick up fresh points promptly).
    """

    UPDATE_INTERVAL = timedelta(minutes=15)

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the forecaster."""
        self._entry = entry
        self._forecast_entity: str = entry.data[CONF_FORECAST_ENTITY]
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

        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{entry.entry_id}_horizon_bias",
            update_interval=self.UPDATE_INTERVAL,
            config_entry=entry,
        )

    @property
    def source_entity(self) -> str:
        """Return the wrapped forecast entity ID (for HAFO sensor compatibility)."""
        return self._forecast_entity

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
        result = await self._generate_forecast()
        _LOGGER.debug(
            "Generated horizon-bias-corrected forecast for %s with %d points",
            self._forecast_entity,
            len(result.forecast),
        )
        return result

    async def _generate_forecast(self) -> ForecastResult:
        """Generate a bias-corrected forecast."""
        now = dt_util.now()
        start_time = now - timedelta(days=self._history_days)

        actual_stats = await _get_statistics(self.hass, self._reference_entity, start_time, now)
        forecast_stats = await _get_statistics(self.hass, self._forecast_entity, start_time, now)

        if not actual_stats or not forecast_stats:
            msg = (
                f"No historical data available to learn bias for "
                f"{self._forecast_entity} vs {self._reference_entity}"
            )
            raise ValueError(msg)

        bias_table = build_bias_table(actual_stats, forecast_stats, now, self._min_days_per_bucket)

        raw_points = get_forecast_points(self.hass, self._forecast_entity)
        if not raw_points:
            msg = f"No forecast points available on {self._forecast_entity}"
            raise ValueError(msg)

        corrected = apply_bias(raw_points, bias_table, now)

        _LOGGER.debug(
            "Bias table for %s covers %d/%d quarter-of-day slots",
            self._forecast_entity,
            len(bias_table),
            QUARTERS_PER_DAY,
        )

        return ForecastResult(
            forecast=corrected,
            source_entity=self._forecast_entity,
            history_days=self._history_days,
            generated_at=now,
        )

    def cleanup(self) -> None:
        """Clean up coordinator resources."""
        _LOGGER.debug("Cleaning up horizon-bias forecaster for %s", self._forecast_entity)
