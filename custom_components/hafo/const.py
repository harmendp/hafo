"""Constants for the Home Assistant Forecaster integration."""

from typing import Final

DOMAIN: Final = "hafo"

# Configuration keys
CONF_SOURCE_ENTITY: Final = "source_entity"
CONF_HISTORY_DAYS: Final = "history_days"
CONF_FORECAST_TYPE: Final = "forecast_type"
CONF_CUSTOM_NAME = "custom_name"

# Source entity attributes (persisted to survive restarts)
CONF_SOURCE_UNIT: Final = "source_unit_of_measurement"
CONF_SOURCE_DEVICE_CLASS: Final = "source_device_class"

# Forecast types
FORECAST_TYPE_HISTORICAL_SHIFT: Final = "historical_shift"
FORECAST_TYPE_HISTORICAL_AVERAGED = "historical_averaged"

# Default values
DEFAULT_HISTORY_DAYS: Final = 7
DEFAULT_FORECAST_TYPE: Final = FORECAST_TYPE_HISTORICAL_SHIFT

# Attribute keys
ATTR_FORECAST: Final = "forecast"
ATTR_LAST_UPDATED: Final = "last_forecast_update"
ATTR_SOURCE_ENTITY: Final = "source_entity"
ATTR_HISTORY_DAYS: Final = "history_days"

CONF_FORECAST_ENTITIES: Final = "forecast_entities"
CONF_REFERENCE_ENTITY: Final = "reference_entity"
FORECAST_TYPE_HORIZON_BIAS: Final = "horizon_bias"
DEFAULT_BIAS_HISTORY_DAYS: Final = 21

# Horizon bias: minimum number of distinct days required in a quarter-of-day
# bucket before the learned correction factor is applied (see horizon_bias.py
# for why this counts days, not raw 5-min samples).
CONF_MIN_DAYS_PER_BUCKET: Final = "min_days_per_bucket"
DEFAULT_MIN_DAYS_PER_BUCKET: Final = 5
