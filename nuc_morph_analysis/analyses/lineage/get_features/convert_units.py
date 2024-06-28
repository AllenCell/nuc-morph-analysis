def convert_duration_to_hours(single_feature_per_track_df, interval):
    """
    Convert duration_BC from frames to hours

    Parameters
    ----------
    single_feature_per_track_df: DataFrame
        DataFrame containing a single metric per track

    interval: float
        time interval in minutes

    Returns
    -------
    single_feature_per_track_df: DataFrame
        DataFrame with duration_BC converted to hours
    """
    single_feature_per_track_df["duration_BC_hr"] = (
        single_feature_per_track_df["duration_BC"] * interval / 60
    )
    return single_feature_per_track_df


def convert_growth_rate_to_per_hour(single_feature_per_track_df, interval):
    """
    Convert growth rate from frames to µm^3 per hours

    Parameters
    ----------
    single_feature_per_track_df: DataFrame
        DataFrame containing a single metric per track

    interval: float
        time interval in minutes

    Returns
    -------
    single_feature_per_track_df: DataFrame
        DataFrame with growth rate converted to hours
    """
    single_feature_per_track_df["late_growth_rate_by_endpoints"] = single_feature_per_track_df[
        "late_growth_rate_by_endpoints"
    ] * (60 / interval)
    return single_feature_per_track_df
