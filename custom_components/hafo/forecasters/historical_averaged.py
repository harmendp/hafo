"""Historical averaged forecaster.

This forecaster builds a forecast by averaging historical statistics from the
past N days (default 28) and projecting that average pattern into the future.

Resolution strategy:
- Past 10 days: 5-minute statistics → average per (weekday, hour, 15-min slot)
- Full history: hourly statistics → average per (weekday, hour) as fallback

The first point of the forecast uses the current sensor state to avoid
forward-filling with stale recorder data.
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

from custom_components.hafo.const import CONF_HISTORY_DAYS, CONF_SOURCE_ENTITY, DEFAULT_HISTORY_DAYS, DOMAIN

_LOGGER = logging.getLogger(__name__)

# Type alias for statistics
StatisticsLike = Mapping[str, Any]

# FIX 1: verhoogd van 7 naar 28 zodat elk weekdag-slot minstens 4 samples heeft
DEFAULT_AVERAGE_DAYS = 28

# Use 5-min stats for the most recent days, hourly for older data
FINE_STATS_DAYS = 10

# Recency weighting: elke dag verder terug wordt het gewicht vermenigvuldigd
# met deze factor. 0.85 betekent: 1 dag oud → 0.85, 7 dagen → 0.32, 28 dagen → 0.01
RECENCY_DECAY = 0.85


async def _get_statistics(
    hass: HomeAssistant,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
    period: str,
) -> list[StatisticsLike]:
    """Fetch statistics for a sensor entity at the given period resolution."""
    recorder = get_instance(hass)
    statistics: dict[str, list[StatisticsRow]] = await recorder.async_add_executor_job(
        lambda: statistics_during_period(
            hass,
            start_time,
            end_time,
            {entity_id},
            period,
            None,
            {"mean"},
        )
    )
    return list(statistics.get(entity_id, []))


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

    MAD is robuuster dan standaarddeviatie omdat uitschieters de MAD zelf
    niet beïnvloeden. Minimaal 4 samples nodig, anders niets weggooien.
    threshold=2.0 staat gelijk aan ~2 mediaan-afwijkingen.
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


def _build_quarter_averages(
    fine_stats: Sequence[StatisticsLike],
    now: datetime,
) -> dict[tuple[int, int, int], float]:
    """Build recency-weighted averages per (weekday, hour, quarter) from 5-min statistics.

    Quarters: 0=:00-:14, 1=:15-:29, 2=:30-:44, 3=:45-:59
    Uitschieters worden per slot gefilterd voor het gewogen gemiddelde.
    """
    slot_samples: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    for stat in fine_stats:
        dt_start = _parse_stat_time(stat.get("start"))
        mean = stat.get("mean")
        if dt_start is None or mean is None:
            continue
        age_days = (now - dt_start).total_seconds() / 86400
        weight = RECENCY_DECAY ** age_days
        quarter = dt_start.minute // 15
        key = (dt_start.weekday(), dt_start.hour, quarter)
        slot_samples.setdefault(key, []).append((float(mean), weight))

    result = {}
    for k, samples in slot_samples.items():
        values = [v for v, _ in samples]
        clean_values = set(_remove_outliers(values))
        filtered = [(v, w) for v, w in samples if v in clean_values]
        if filtered:
            result[k] = sum(v * w for v, w in filtered) / sum(w for _, w in filtered)
    return result


def _build_hour_averages(
    hourly_stats: Sequence[StatisticsLike],
    now: datetime,
) -> dict[tuple[int, int], float]:
    """Build recency-weighted averages per (weekday, hour) from hourly statistics.

    Uitschieters worden per slot gefilterd voor het gewogen gemiddelde.
    """
    slot_samples: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for stat in hourly_stats:
        dt_start = _parse_stat_time(stat.get("start"))
        mean = stat.get("mean")
        if dt_start is None or mean is None:
            continue
        age_days = (now - dt_start).total_seconds() / 86400
        weight = RECENCY_DECAY ** age_days
        key = (dt_start.weekday(), dt_start.hour)
        slot_samples.setdefault(key, []).append((float(mean), weight))

    result = {}
    for k, samples in slot_samples.items():
        values = [v for v, _ in samples]
        clean_values = set(_remove_outliers(values))
        filtered = [(v, w) for v, w in samples if v in clean_values]
        if filtered:
            result[k] = sum(v * w for v, w in filtered) / sum(w for _, w in filtered)
    return result


def build_forecast(
    quarter_averages: dict[tuple[int, int, int], float],
    hour_averages: dict[tuple[int, int], float],
    forecast_days: int = 3,
) -> list[ForecastPoint]:
    """Build a 15-min forecast combining quarter and hourly averages.

    For each 15-min slot:
    1. Use quarter_averages (from 5-min stats) if available
    2. Fall back to hour_averages (from hourly stats)
    3. Fall back to average of same hour across all weekdays
    """
    now = dt_util.now()
    # Start at next 15-min boundary
    current_quarter_start = now.replace(
        minute=(now.minute // 15) * 15, second=0, microsecond=0
    )
    next_quarter = current_quarter_start + timedelta(minutes=15)

    forecast: list[ForecastPoint] = []
    total_quarters = forecast_days * 24 * 4

    for q_offset in range(total_quarters):
        future_time = next_quarter + timedelta(minutes=q_offset * 15)
        weekday = future_time.weekday()
        hour = future_time.hour
        quarter = future_time.minute // 15

        # 1. Fine-grained average
        value = quarter_averages.get((weekday, hour, quarter))

        # 2. Hourly average
        if value is None:
            value = hour_averages.get((weekday, hour))

        # 3. Same hour, any weekday
        if value is None:
            same_hour = [v for (wd, h), v in hour_averages.items() if h == hour]
            value = sum(same_hour) / len(same_hour) if same_hour else None

        if value is not None:
            forecast.append(ForecastPoint(time=future_time, value=value))

    return forecast


class HistoricalAveragedForecaster(DataUpdateCoordinator[ForecastResult | None]):
    """Forecaster that builds forecasts by averaging historical statistics.

    Uses 5-min statistics for recent days (quarter-hour resolution) and
    hourly statistics for the full history window. Groups data by
    (weekday, hour[, quarter]) to respect day-of-week patterns and smooth
    out outliers.

    Update interval: 15 minutes.
    """

    UPDATE_INTERVAL = timedelta(minutes=15)

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the forecaster."""
        self._entry = entry
        self._source_entity: str = entry.data[CONF_SOURCE_ENTITY]
        self._history_days: int = int(
            entry.options.get(CONF_HISTORY_DAYS, entry.data.get(CONF_HISTORY_DAYS, DEFAULT_AVERAGE_DAYS))
        )

        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{entry.entry_id}_averaged",
            update_interval=self.UPDATE_INTERVAL,
            config_entry=entry,
        )

    @property
    def source_entity(self) -> str:
        """Return the source entity ID."""
        return self._source_entity

    @property
    def history_days(self) -> int:
        """Return the number of history days used for forecasting."""
        return self._history_days

    @property
    def entry(self) -> ConfigEntry:
        """Return the config entry."""
        return self._entry

    async def _async_update_data(self) -> ForecastResult:
        """Fetch and update forecast data."""
        result = await self._generate_forecast()
        _LOGGER.debug(
            "Generated averaged forecast for %s with %d points",
            self._source_entity,
            len(result.forecast),
        )
        return result

    async def _generate_forecast(self) -> ForecastResult:
        """Generate a forecast by averaging historical data."""
        now = dt_util.now()

        # Fetch fine-grained (5-min) stats for recent days
        fine_days = min(FINE_STATS_DAYS, self._history_days)
        fine_stats = await _get_statistics(
            self.hass,
            self._source_entity,
            start_time=now - timedelta(days=fine_days),
            end_time=now,
            period="5minute",
        )

        # FIX 2: altijd uurlijkse stats ophalen over de volledige geschiedenis,
        # niet alleen als history_days > fine_days. Dit zorgt voor een betrouwbare
        # fallback met voldoende samples ook bij korte history_days.
        hourly_stats = await _get_statistics(
            self.hass,
            self._source_entity,
            start_time=now - timedelta(days=self._history_days),
            end_time=now,
            period="hour",
        )

        if not fine_stats and not hourly_stats:
            msg = f"No historical data available for {self._source_entity}"
            raise ValueError(msg)

        quarter_averages = _build_quarter_averages(fine_stats, now)
        hour_averages = _build_hour_averages(hourly_stats, now)

        # Supplement hour_averages with hourly aggregates from fine_stats
        # so the fallback always has data even without old hourly stats
        fine_hour_samples: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for stat in fine_stats:
            dt_start = _parse_stat_time(stat.get("start"))
            mean = stat.get("mean")
            if dt_start is None or mean is None:
                continue
            age_days = (now - dt_start).total_seconds() / 86400
            weight = RECENCY_DECAY ** age_days
            key = (dt_start.weekday(), dt_start.hour)
            fine_hour_samples.setdefault(key, []).append((float(mean), weight))
        for key, samples in fine_hour_samples.items():
            if key not in hour_averages:
                values = [v for v, _ in samples]
                clean_values = set(_remove_outliers(values))
                filtered = [(v, w) for v, w in samples if v in clean_values]
                if filtered:
                    hour_averages[key] = sum(v * w for v, w in filtered) / sum(w for _, w in filtered)

        forecast = build_forecast(quarter_averages, hour_averages)

        if not forecast:
            msg = f"No valid forecast points generated for {self._source_entity}"
            raise ValueError(msg)

        # Prepend current sensor state to cover the gap between now and the
        # first forecast point (recorder lags behind by up to 15 minutes).
        current_state = self.hass.states.get(self._source_entity)
        if current_state is not None:
            try:
                current_value = float(current_state.state)
                current_point = ForecastPoint(time=now, value=current_value)
                if not forecast or now < forecast[0].time:
                    forecast = [current_point, *forecast]
            except (ValueError, TypeError):
                pass

        return ForecastResult(
            forecast=forecast,
            source_entity=self._source_entity,
            history_days=self._history_days,
            generated_at=now,
        )

    def cleanup(self) -> None:
        """Clean up coordinator resources."""
        _LOGGER.debug("Cleaning up averaged forecaster for %s", self._source_entity)
