"""Business vocabulary shared by tools, seed data, and graders."""

from enum import StrEnum


class Trade(StrEnum):
    HVAC = "hvac"
    PLUMBING = "plumbing"


class Service(StrEnum):
    FURNACE_REPAIR = "furnace_repair"
    BOILER_REPAIR = "boiler_repair"
    AC_REPAIR = "ac_repair"
    HEAT_PUMP_REPAIR = "heat_pump_repair"
    HVAC_MAINTENANCE = "hvac_maintenance"
    HVAC_REPLACEMENT_ESTIMATE = "hvac_replacement_estimate"
    WATER_HEATER_REPAIR = "water_heater_repair"
    WATER_HEATER_REPLACEMENT_ESTIMATE = "water_heater_replacement_estimate"
    DRAIN_CLOG = "drain_clog"
    LEAK_REPAIR = "leak_repair"
    TOILET_REPAIR = "toilet_repair"


SERVICE_TRADE: dict[Service, Trade] = {
    Service.FURNACE_REPAIR: Trade.HVAC,
    Service.BOILER_REPAIR: Trade.HVAC,
    Service.AC_REPAIR: Trade.HVAC,
    Service.HEAT_PUMP_REPAIR: Trade.HVAC,
    Service.HVAC_MAINTENANCE: Trade.HVAC,
    Service.HVAC_REPLACEMENT_ESTIMATE: Trade.HVAC,
    Service.WATER_HEATER_REPAIR: Trade.PLUMBING,
    Service.WATER_HEATER_REPLACEMENT_ESTIMATE: Trade.PLUMBING,
    Service.DRAIN_CLOG: Trade.PLUMBING,
    Service.LEAK_REPAIR: Trade.PLUMBING,
    Service.TOILET_REPAIR: Trade.PLUMBING,
}


# Price policy. The agent may only quote what get_pricing returns (see SOPs).
PRICING: dict[Service, dict] = {
    Service.FURNACE_REPAIR: {
        "diagnostic_fee": 89,
        "note": "Diagnostic fee waived if repair is done.",
    },
    Service.BOILER_REPAIR: {
        "diagnostic_fee": 89,
        "note": "Diagnostic fee waived if repair is done.",
    },
    Service.AC_REPAIR: {"diagnostic_fee": 89, "note": "Diagnostic fee waived if repair is done."},
    Service.HEAT_PUMP_REPAIR: {
        "diagnostic_fee": 89,
        "note": "Diagnostic fee waived if repair is done.",
    },
    Service.HVAC_MAINTENANCE: {"flat_price": 149, "note": "Members of the comfort club pay $0."},
    Service.HVAC_REPLACEMENT_ESTIMATE: {
        "flat_price": 0,
        "note": "Free in-home estimate. Never quote replacement prices on the phone.",
    },
    Service.WATER_HEATER_REPAIR: {
        "diagnostic_fee": 79,
        "note": "Diagnostic fee waived if repair is done.",
    },
    Service.WATER_HEATER_REPLACEMENT_ESTIMATE: {
        "flat_price": 0,
        "note": "Free in-home estimate. Never quote replacement prices on the phone.",
    },
    Service.DRAIN_CLOG: {
        "flat_price": 189,
        "note": "Main-line clogs may cost more after inspection.",
    },
    Service.LEAK_REPAIR: {"diagnostic_fee": 79, "note": "Diagnostic fee waived if repair is done."},
    Service.TOILET_REPAIR: {
        "diagnostic_fee": 79,
        "note": "Diagnostic fee waived if repair is done.",
    },
}

EMERGENCY_FEE = 150  # added to same-day emergency visits
