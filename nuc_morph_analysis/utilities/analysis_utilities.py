import numpy as np
from scipy import stats


def get_correlation_values(x_values, y_values, correlation_metric="pearson", bootstrap_count=0):
    """
    Calculates correlation between two vectors of values

    Parameters
    ----------
    x_values: np.array
        Vector of values
    y_values: np.array
        Vector of values
    correlation_metric: str
        Metric to use for correlation calculation
        Available options are "pearson", "spearman"
    bootstrap_count: int
        Number of bootstrap iterations to perform.
        If 0, no bootstrapping is performed

    Returns
    ----------
    correlation: float
        Correlation value
    p_value: float
        P value for correlation. Average of bootstrapped p values is returned
        if bootstrapping is performed.
    error: float
        Standard error in correlation value calculated from bootstrapping
    ci: float
        Confidence interval calculated from bootstrapping
    """
    if bootstrap_count == 0:
        if correlation_metric == "pearson":
            correlation, p_value = stats.pearsonr(x_values, y_values)
        elif correlation_metric == "spearman":
            correlation, p_value = stats.spearmanr(x_values, y_values)
        else:
            raise ValueError(
                f"Correlation metric {correlation_metric} not supported. Please use 'pearson' or 'spearman'"
            )
        error = None
        ci = None
    else:
        corr_vals = np.zeros(bootstrap_count)
        p_values = np.zeros(bootstrap_count)
        for ct in range(bootstrap_count):
            inds = np.random.choice(np.arange(len(x_values)), size=len(x_values))
            x_values_boot = x_values[inds]
            y_values_boot = y_values[inds]
            if correlation_metric == "pearson":
                corr_val, p_val = stats.pearsonr(x_values_boot, y_values_boot)
            elif correlation_metric == "spearman":
                corr_val, p_val = stats.spearmanr(x_values_boot, y_values_boot)
            else:
                raise ValueError(
                    f"Correlation metric {correlation_metric} not supported. Please use 'pearson' or 'spearman'"
                )
            corr_vals[ct] = corr_val
            p_values[ct] = p_val
        correlation = np.nanmean(corr_vals)
        p_value = np.nanmean(p_values)
        error = np.nanstd(corr_vals) / np.sqrt(bootstrap_count)
        ci = np.percentile(corr_vals, [5, 95])

    return correlation, p_value, error, ci
