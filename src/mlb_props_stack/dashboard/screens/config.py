"""Config screen renderer."""

from __future__ import annotations

from dataclasses import asdict

from ..lib.data import DashboardSettings


def render_config_screen(
    *,
    st: object,
    settings: DashboardSettings,
) -> tuple[DashboardSettings, bool]:
    """Render the config screen and return the updated settings."""
    st.markdown(
        "<div class='strike-screen-head'>"
        "<div><div class='strike-screen-title'>CONFIG "
        "<span class='strike-dim'>/ thresholds · bankroll · model</span></div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(2, gap="large")
    with columns[0]:
        edge_min = st.number_input(
            "Min edge (%)",
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            value=float(settings.edge_min * 100.0),
            key="cfg_edge_min_pct",
        )
        confidence_min = st.number_input(
            "Min confidence",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(settings.confidence_min),
            key="cfg_confidence_min",
        )
        devig_method = st.selectbox(
            "Vig removal",
            ["shin", "power", "multiplicative"],
            index=["shin", "power", "multiplicative"].index(settings.devig_method),
            key="cfg_devig_method",
        )
        max_hold = st.number_input(
            "Max hold (%)",
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            value=float(settings.max_hold * 100.0),
            key="cfg_max_hold_pct",
        )
        kelly_fraction = st.number_input(
            "Kelly fraction",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=float(settings.kelly_fraction),
            key="cfg_kelly_fraction",
        )
        max_stake_units = st.number_input(
            "Max stake / prop (u)",
            min_value=0.0,
            max_value=25.0,
            step=0.25,
            value=float(settings.max_stake_units),
            key="cfg_max_stake_units",
        )
        max_daily_exposure_units = st.number_input(
            "Max daily exposure (u)",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=float(settings.max_daily_exposure_units),
            key="cfg_max_daily_exposure_units",
        )
    with columns[1]:
        bankroll_units = st.number_input(
            "Bankroll (u)",
            min_value=1.0,
            max_value=10000.0,
            step=10.0,
            value=float(settings.bankroll_units),
            key="cfg_bankroll_units",
        )
        calibration_method = st.selectbox(
            "Calibration",
            ["isotonic", "platt", "beta", "none"],
            index=["isotonic", "platt", "beta", "none"].index(settings.calibration_method),
            key="cfg_calibration_method",
        )
        retrain_window_days = st.number_input(
            "Retrain window (days)",
            min_value=1,
            max_value=3650,
            step=5,
            value=int(settings.retrain_window_days),
            key="cfg_retrain_window_days",
        )
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=100000,
            step=1,
            value=int(settings.seed),
            key="cfg_seed",
        )
        decision_cutoff_minutes = st.number_input(
            "Decision cutoff (minutes before first pitch)",
            min_value=0,
            max_value=360,
            step=5,
            value=int(settings.decision_cutoff_minutes),
            key="cfg_decision_cutoff_minutes",
        )
        refresh_cadence_minutes = st.number_input(
            "Refresh cadence (minutes)",
            min_value=1,
            max_value=120,
            step=1,
            value=int(settings.refresh_cadence_minutes),
            key="cfg_refresh_cadence_minutes",
        )
        active_run_id = st.text_input(
            "Active model run id (optional override)",
            value=settings.active_run_id or "",
            key="cfg_active_run_id",
        ).strip() or None

    updated = DashboardSettings(
        edge_min=edge_min / 100.0,
        confidence_min=confidence_min,
        devig_method=devig_method,
        max_hold=max_hold / 100.0,
        kelly_fraction=kelly_fraction,
        max_stake_units=max_stake_units,
        max_daily_exposure_units=max_daily_exposure_units,
        bankroll_units=bankroll_units,
        calibration_method=calibration_method,
        retrain_window_days=int(retrain_window_days),
        seed=int(seed),
        decision_cutoff_minutes=int(decision_cutoff_minutes),
        refresh_cadence_minutes=int(refresh_cadence_minutes),
        active_run_id=active_run_id,
        active_model_label=settings.active_model_label,
    )
    save_clicked = st.button("Persist to user_config.toml", key="cfg_save")
    st.caption("Config updates apply to the current session immediately. Persist saves them for later runs.")
    return updated, save_clicked
