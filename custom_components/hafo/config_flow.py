"""Config flow for Home Assistant Forecaster integration."""

from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers import selector
import voluptuous as vol

from .const import (
    CONF_FORECAST_ENTITY,
    CONF_FORECAST_TYPE,
    CONF_HISTORY_DAYS,
    CONF_MIN_DAYS_PER_BUCKET,
    CONF_REFERENCE_ENTITY,
    CONF_SOURCE_ENTITY,
    DEFAULT_BIAS_HISTORY_DAYS,
    DEFAULT_FORECAST_TYPE,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_MIN_DAYS_PER_BUCKET,
    DOMAIN,
    FORECAST_TYPE_HISTORICAL_AVERAGED,
    FORECAST_TYPE_HISTORICAL_SHIFT,
    FORECAST_TYPE_HORIZON_BIAS,
)

# Forecast types that use the single generic CONF_SOURCE_ENTITY field
# rather than the forecast_entity/reference_entity pair.
_SOURCE_ENTITY_TYPES = {FORECAST_TYPE_HISTORICAL_SHIFT, FORECAST_TYPE_HISTORICAL_AVERAGED}


class HafoConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Assistant Forecaster."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> "HafoOptionsFlow":  # noqa: ARG004
        """Get the options flow for this handler."""
        return HafoOptionsFlow()

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            forecast_type = user_input.get(CONF_FORECAST_TYPE, DEFAULT_FORECAST_TYPE)

            if forecast_type in _SOURCE_ENTITY_TYPES:
                # Historical Shift / Historical Averaged: single generic entity.
                source_entity = user_input.get(CONF_SOURCE_ENTITY)
                if not source_entity:
                    errors[CONF_SOURCE_ENTITY] = "required"
                elif self.hass.states.get(source_entity) is None:
                    errors[CONF_SOURCE_ENTITY] = "entity_not_found"
                else:
                    unique_key = source_entity
            else:
                # Horizon Bias: needs both the forecast entity to correct and
                # the actual-production entity to learn the correction from.
                forecast_entity = user_input.get(CONF_FORECAST_ENTITY)
                reference_entity = user_input.get(CONF_REFERENCE_ENTITY)
                if not forecast_entity:
                    errors[CONF_FORECAST_ENTITY] = "required"
                elif self.hass.states.get(forecast_entity) is None:
                    errors[CONF_FORECAST_ENTITY] = "entity_not_found"
                if not reference_entity:
                    errors[CONF_REFERENCE_ENTITY] = "required"
                elif self.hass.states.get(reference_entity) is None:
                    errors[CONF_REFERENCE_ENTITY] = "entity_not_found"
                unique_key = f"{forecast_entity}_{reference_entity}"

            if not errors:
                # Create unique ID from the relevant entity/entities
                await self.async_set_unique_id(f"{DOMAIN}_{unique_key}")
                self._abort_if_unique_id_configured()

                # Create a friendly title from the primary entity
                title_entity = user_input.get(CONF_SOURCE_ENTITY) or user_input.get(CONF_FORECAST_ENTITY)
                state = self.hass.states.get(title_entity) if title_entity else None
                title = state.attributes.get("friendly_name", title_entity) if state else title_entity

                return self.async_create_entry(
                    title=title,
                    data=user_input,
                )

        # Build the schema for user input. All fields are shown at once (no
        # multi-step wizard); only the ones relevant to the chosen
        # forecast_type need to be filled in, the rest can stay blank.
        submitted_type = (user_input or {}).get(CONF_FORECAST_TYPE, DEFAULT_FORECAST_TYPE)
        default_history_days = DEFAULT_BIAS_HISTORY_DAYS if submitted_type == FORECAST_TYPE_HORIZON_BIAS else DEFAULT_HISTORY_DAYS

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_FORECAST_TYPE,
                    default=DEFAULT_FORECAST_TYPE,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(
                                value=FORECAST_TYPE_HISTORICAL_SHIFT,
                                label="Historical Shift (single day)",
                            ),
                            selector.SelectOptionDict(
                                value=FORECAST_TYPE_HISTORICAL_AVERAGED,
                                label="Historical Averaged (weekday pattern)",
                            ),
                            selector.SelectOptionDict(
                                value=FORECAST_TYPE_HORIZON_BIAS,
                                label="Horizon Bias (correct another forecast for local shading)",
                            ),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_SOURCE_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor", "input_number"])
                ),
                vol.Optional(CONF_FORECAST_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="power")
                ),
                vol.Optional(CONF_REFERENCE_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="power")
                ),
                vol.Optional(
                    CONF_MIN_DAYS_PER_BUCKET,
                    default=DEFAULT_MIN_DAYS_PER_BUCKET,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=4,
                        max=30,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="days",
                    )
                ),
                vol.Optional(
                    CONF_HISTORY_DAYS,
                    default=default_history_days,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=60,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="days",
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "hint": (
                    "Historical Shift/Averaged: fill in 'Source entity' only. "
                    "Horizon Bias: fill in 'Forecast entity' (the forecast to correct) "
                    "and 'Reference entity' (actual measured production), leave 'Source entity' empty."
                )
            },
        )


class HafoOptionsFlow(OptionsFlow):
    """Handle options flow for Home Assistant Forecaster."""

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle options flow."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Get current config
        entry = self.hass.config_entries.async_get_entry(self.handler)  # type: ignore[arg-type]
        if entry is None:
            return self.async_abort(reason="entry_not_found")

        # Read from options first (previously saved), fall back to data (initial config)
        current_history_days = entry.options.get(
            CONF_HISTORY_DAYS, entry.data.get(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS)
        )

        schema_dict: dict[Any, Any] = {
            vol.Optional(
                CONF_HISTORY_DAYS,
                default=current_history_days,
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=60,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement="days",
                )
            ),
        }

        if entry.data.get(CONF_FORECAST_TYPE) == FORECAST_TYPE_HORIZON_BIAS:
            current_min_days = entry.options.get(
                CONF_MIN_DAYS_PER_BUCKET,
                entry.data.get(CONF_MIN_DAYS_PER_BUCKET, DEFAULT_MIN_DAYS_PER_BUCKET),
            )
            schema_dict[
                vol.Optional(CONF_MIN_DAYS_PER_BUCKET, default=current_min_days)
            ] = selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=4,
                    max=30,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement="days",
                )
            )

        schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
        )
