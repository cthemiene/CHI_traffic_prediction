"""
Time series forecasting for parking occupancy.

This module provides utilities to aggregate historical parking occupancy
observations and forecast future demand within a given operating window.  It
supports two forecasting back‑ends: Facebook's Prophet (if installed) and
statsmodels' ARIMA/SARIMAX.  Prophet has been shown to provide competitive
results for parking occupancy forecasting【229841028231096†L1055-L1105】 but is not
available in this environment, so a fallback to ARIMA is provided.

The main class ``OccupancyForecaster`` loads aggregated occupancy data from a
CSV or list and trains the appropriate model.  Predictions are returned as
pandas DataFrames with timestamps and forecasted occupancy ratios.

Best practices for forecasting include:

* Aggregating raw observations into regular intervals (e.g., 15 minutes) to
  smooth noise and align with human‑meaningful time slots.
* Accounting for daily seasonality and weekly patterns.  Prophet and SARIMAX
  both model seasonality; Prophet automatically handles holidays and change
  points【229841028231096†L1055-L1105】.
* Evaluating multiple models on historical data to select the best performer.
  Research on urban parking occupancy forecasting showed that Prophet and
  Neural Prophet outperformed LSTM and SARIMAX for several datasets【229841028231096†L1055-L1105】.

"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import pandas as pd  # type: ignore
from zoneinfo import ZoneInfo

try:
    # Prophet is not installed by default; import if available
    from prophet import Prophet  # type: ignore
except ImportError:
    Prophet = None  # type: ignore

from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore


@dataclass
class Observation:
    """Simple container for a single occupancy observation.

    Attributes
    ----------
    timestamp : datetime
        The time at which the observation was recorded (timezone aware).
    ratio : float
        Occupancy ratio, defined as ``n_occupied / total_spaces``.  Must be in
        [0, 1].
    """

    timestamp: dt.datetime
    ratio: float


class OccupancyForecaster:
    """Forecast occupancy using Prophet or ARIMA.

    This class provides methods to ingest observations, fit a forecasting
    model, and make predictions for future time ranges.  It allows specifying
    operating hours to restrict predictions to times when the facility is
    open (e.g., 07:00–19:00 America/Chicago).
    """

    def __init__(
        self,
        timezone: str = "America/Chicago",
        operating_hours: Tuple[int, int] = (7, 19),
        resample_minutes: int = 15,
        use_prophet: bool = False,
    ) -> None:
        """Initialize the forecaster.

        Parameters
        ----------
        timezone : str, optional
            IANA timezone string for the parking lot.  All timestamps are
            localized to this zone.  Defaults to "America/Chicago".
        operating_hours : tuple of int, optional
            Start and end hour (24‑hr) of the facility's operation.  Only
            predictions during this window are considered for the "best time"
            recommendation.  Defaults to (7, 19) for 7 AM to 7 PM.
        resample_minutes : int, optional
            Resample interval for aggregating occupancy ratios.  Defaults to
            15 minutes.
        use_prophet : bool, optional
            Whether to use Prophet if installed.  If True but Prophet is not
            available, falls back to SARIMAX with a warning.  If False,
            SARIMAX is always used.
        """
        self.tz = ZoneInfo(timezone)
        self.operating_hours = operating_hours
        self.resample_minutes = resample_minutes
        self.use_prophet = use_prophet and (Prophet is not None)
        self._model = None
        self._fitted = False

    def load_observations(self, observations: Iterable[Observation]) -> pd.DataFrame:
        """Convert a sequence of observations into a resampled DataFrame.

        Parameters
        ----------
        observations : iterable of Observation
            Raw occupancy observations.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by timestamp with a single column ``ratio``
            containing resampled occupancy ratios.  Missing intervals are
            forward filled to maintain continuity.
        """
        # Convert to DataFrame
        records = [(obs.timestamp.astimezone(self.tz), obs.ratio) for obs in observations]
        df = pd.DataFrame(records, columns=["timestamp", "ratio"]).set_index("timestamp")
        # Resample to regular intervals
        df_resampled = (
            df.resample(f"{self.resample_minutes}min").mean().interpolate(method="time")
        )
        return df_resampled

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the forecasting model to resampled data.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame produced by ``load_observations`` containing the
            ``ratio`` column indexed by timestamp.  The index must be
            monotonic and timezone aware.
        """
        if self.use_prophet:
            # Prophet expects columns named 'ds' and 'y'
            prophet_df = df.reset_index().rename(columns={"timestamp": "ds", "ratio": "y"})
            self._model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            self._model.fit(prophet_df)  # type: ignore[arg-type]
            self._fitted = True
        else:
            # Fit a SARIMAX model with seasonal order.  We choose (1, 0, 1) with
            # daily seasonality (24*60/resample_minutes).  Users may tune this.
            period = int((24 * 60) / self.resample_minutes)
            # Statsmodels requires integer indices; convert to pandas RangeIndex
            y = df["ratio"].tolist()
            self._model = SARIMAX(
                y,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._model = self._model.fit(disp=False)
            self._index_start_time = df.index[0]
            self._fitted = True

    def forecast(self, horizon_hours: int = 12) -> pd.DataFrame:
        """Generate a forecast for the next ``horizon_hours`` hours.

        Parameters
        ----------
        horizon_hours : int, optional
            Number of hours into the future to forecast.  Defaults to 12.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``timestamp`` and ``ratio_pred`` containing
            predicted occupancy ratios for each resampled interval.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting.")
        if self.use_prophet:
            # Create future dataframe with desired horizon
            periods = int(horizon_hours * 60 / self.resample_minutes)
            future = self._model.make_future_dataframe(periods=periods, freq=f"{self.resample_minutes}min")
            forecast = self._model.predict(future)
            forecast = forecast[["ds", "yhat"]].tail(periods)
            forecast = forecast.rename(columns={"ds": "timestamp", "yhat": "ratio_pred"})
            forecast["timestamp"] = pd.to_datetime(forecast["timestamp"]).dt.tz_localize(self.tz)
            return forecast
        else:
            steps = int(horizon_hours * 60 / self.resample_minutes)
            pred = self._model.get_forecast(steps)
            ratios = pred.predicted_mean
            index = [
                self._index_start_time + dt.timedelta(minutes=self.resample_minutes * (i + 1))
                for i in range(steps)
            ]
            ts = pd.to_datetime(index).tz_localize(self.tz)
            return pd.DataFrame({"timestamp": ts, "ratio_pred": ratios})

    def best_time_to_visit(self, forecast_df: pd.DataFrame, date: Optional[dt.date] = None) -> Tuple[dt.datetime, float]:
        """Return the timestamp with the lowest predicted occupancy within operating hours.

        Parameters
        ----------
        forecast_df : pandas.DataFrame
            Forecast generated by ``forecast()``.
        date : datetime.date, optional
            Specific date to restrict the search.  If None, uses the date of
            the first forecast entry.

        Returns
        -------
        tuple
            (timestamp, predicted_ratio) for the lowest occupancy within
            operating hours.
        """
        df = forecast_df.copy()
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        if date is None:
            date = df.iloc[0]["date"]
        df_day = df[df["date"] == date]
        start_hour, end_hour = self.operating_hours
        df_day = df_day[(df_day["hour"] >= start_hour) & (df_day["hour"] < end_hour)]
        if df_day.empty:
            raise ValueError("No forecast data available within operating hours for the given date.")
        idx = df_day["ratio_pred"].idxmin()
        row = df_day.loc[idx]
        return row["timestamp"], float(row["ratio_pred"])