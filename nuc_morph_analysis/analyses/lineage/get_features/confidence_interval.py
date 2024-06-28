import numpy as np


def confidence_interval(df, n_resamples, column_2, column_3):
    """
    Bootstrap calculate the confidence interval for a correlation between two columns in a pandas dataframe

    Parameters
    ----------
    df: Dataframe
        The subset dataframe containing the columns you want to compare
    n_resamples: Int
        Number of times to resample the data
    column_2: String
        Dataframe column name to compare (ie. 'tid1_late_gr')
    column_3: String
        Dataframe column name to compare (ie. 'tid2_late_gr')

    Returns
    -------
    mean: Float
        Mean of the resampled correlations
    std_dev: Float
        Standard deviation of the resampled correlations
    re_sampled_corr: List
        List containing all the correlation values calculated for all the samples for easy visualization in a histogram.
    """
    df = df.reset_index()
    original_data = list(range(0, len(df)))
    re_sampled_corr = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_resamples):
        rints = rng.integers(low=0, high=len(original_data), size=len(original_data))
        df_resampled = df.iloc[rints]
        corr_sample = df_resampled[column_2].corr(df_resampled[column_3], method="pearson")
        re_sampled_corr.append(corr_sample)
    mean = np.mean(re_sampled_corr)
    std_dev = np.std(re_sampled_corr)
    percent = np.percentile(re_sampled_corr, [5, 95])
    return mean, std_dev, re_sampled_corr, percent
