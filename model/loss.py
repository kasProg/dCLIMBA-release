import torch
from scipy.stats import wasserstein_distance
import numpy as np
import torch.nn.functional as F

def interpolate_quantiles(data, device, quantile_levels, num_quantiles=200):
    """
    Interpolates the quantiles of a given dataset to match the desired number of quantile levels.
    """
    sorted_data = torch.sort(data, dim=0)[0]
    data_quantiles = torch.quantile(sorted_data, quantile_levels, dim=0)  # Interpolated quantiles
    return data_quantiles


def distributional_loss_interpolated(transformed_x, target_y, device, num_quantiles=100, emph_quantile=0.5):
    """
    Calculates the distributional loss by interpolating the quantiles of both distributions,
    even if the lengths of the two datasets are different.
    """    
    quantile_levels = torch.linspace(0.0, 1.0, num_quantiles).to(target_y.dtype).to(device)

    # Interpolate the quantiles for both distributions
    quantiles_x = interpolate_quantiles(transformed_x, device, quantile_levels, num_quantiles)
    quantiles_y = interpolate_quantiles(target_y, device, quantile_levels, num_quantiles)
    
    if isinstance(emph_quantile, (float, int)):
        # Single quantile emphasis
        weights = torch.exp(-torch.abs(quantile_levels - emph_quantile))
        weights = weights.view(-1, *[1]*(quantiles_x.dim()-1)).expand_as(quantiles_x)

    elif isinstance(emph_quantile, (list, tuple)) and all(isinstance(q, (float, int)) for q in emph_quantile):
        # Multiple quantiles emphasis
        weights = sum(torch.exp(-torch.abs(quantile_levels - q)) for q in emph_quantile)
        weights = weights.view(-1, *[1]*(quantiles_x.dim()-1)).expand_as(quantiles_x)

    else:
        # No emphasis (uniform weighting)
        weights = 1
        

    # Calculate the mean squared error (MSE) between quantiles
    loss = torch.mean(weights*(quantiles_x - quantiles_y) ** 2)
    return loss

def trend_loss(transformed_x, original_x, device):
    time_index = torch.arange(original_x.size(0), dtype=original_x.dtype).unsqueeze(1).to(device)
    original_trend = torch.linalg.lstsq(time_index, original_x).solution.squeeze()
    transformed_trend = torch.linalg.lstsq(time_index, transformed_x).solution.squeeze()

    trend_loss = torch.mean((transformed_trend - original_trend) ** 2)

    return trend_loss


def rainy_day_loss(transformed_x, target_y, threshold=1):
    """
    Calculates a differentiable loss that targets preserving the number of rainy days in the transformed data.
    """
    # # Use a sigmoid approximation for the "rainy days" indicator
    rainy_indicator_transformed = torch.sigmoid(transformed_x - threshold)
    rainy_indicator_target = torch.sigmoid(target_y - threshold)

    # Binary "rainy day" approximation using ReLU
    # rainy_indicator_transformed = torch.relu((transformed_x - threshold) / threshold)
    # rainy_indicator_target = torch.relu((target_y - threshold) / threshold)

    # Sum over the approximated rainy days in both transformed and target data
    rainy_days_transformed = rainy_indicator_transformed.sum(dim=0)
    rainy_days_target = rainy_indicator_target.sum(dim=0)

    # Calculate the mean absolute difference in the number of rainy days across all series
    rainy_days_loss = torch.mean(torch.abs(rainy_days_transformed - rainy_days_target))
    return rainy_days_loss

def compare_distributions(transformed_x, x, y):
    # Assuming transformed_x, x, and y are 2D tensors of shape (time_steps, num_time_series)
    num_series = transformed_x.shape[1]
    wasserstein_distances = []

    for i in range(num_series):
        # Calculate Wasserstein distance between transformed_x and y
        dist_transformed = wasserstein_distance(transformed_x[:,i], y[:,i])

        # Calculate Wasserstein distance between x and y
        dist_original = wasserstein_distance(x[:, i], y[:,i])

        # Calculate improvement ratio
        improvement_ratio = (dist_original - dist_transformed) / dist_original

        wasserstein_distances.append(improvement_ratio)

    # Average improvement across all series
    avg_improvement = sum(wasserstein_distances) / num_series

    return avg_improvement, wasserstein_distances

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2, axis=0))

def kl_divergence_loss(transformed_x, target_y, num_bins=50):

    prob_log_x = F.log_softmax(transformed_x, dim=1)
    prob_y = F.softmax(target_y, dim=1)

    # Calculate KL divergence for each coordinate and take the mean
    kl_loss = F.kl_div(prob_log_x, prob_y, reduction='batchmean')

    # Return the average KL divergence across all coordinates
    return kl_loss



def wasserstein_distance_loss(predicted, target, dim=1):
    """
    Compute the Wasserstein distance loss between two distributions.

    Args:
        predicted (torch.Tensor): Predicted distribution (logits or probabilities).
        target (torch.Tensor): Target distribution (logits or probabilities).
        dim (int): Dimension along which to compute the distributions.
        
    Returns:
        torch.Tensor: Wasserstein distance loss.
    """
    # Normalize logits to probabilities using softmax if needed
    prob_predicted = torch.softmax(predicted, dim=dim)
    prob_target = torch.softmax(target, dim=dim)

    # Compute the CDFs
    cdf_predicted = torch.cumsum(prob_predicted, dim=dim)
    cdf_target = torch.cumsum(prob_target, dim=dim)

    # Wasserstein distance is the sum of absolute differences between CDFs
    wasserstein_loss = torch.sum(torch.abs(cdf_predicted - cdf_target), dim=dim)
    
    # Average over the batch
    return wasserstein_loss.mean()



def autocorrelation_loss(pred, target, lags=[1, 2, 3, 5]):
    """
    Computes MSE between autocorrelation of predicted and target series for multiple lags.

    Args:
        pred: (batch, time) or (time,) — predicted time series
        target: same shape — observed time series
        lags: list of int — lags at which to compute autocorrelation

    Returns:
        scalar loss: mean squared error over autocorrelations at given lags
    """
    def compute_acf(x, lag):
        # x: (batch, time) or (time,)
        x = x - x.mean(dim=-1, keepdim=True)  # mean-center
        n = x.size(-1)
        if x.dim() == 1:
            return F.cosine_similarity(x[:-lag], x[lag:], dim=0)
        else:
            return F.cosine_similarity(x[:, :-lag], x[:, lag:], dim=1)

    loss = 0.0
    for lag in lags:
        acf_pred = compute_acf(pred, lag)
        acf_target = compute_acf(target, lag)
        loss += F.mse_loss(acf_pred, acf_target)

    return loss / len(lags)

def fourier_spectrum_loss(pred, target):
    seq_len = min(pred.size(1), target.size(1))
    pred = pred[:, :seq_len]
    target = target[:, :seq_len]

    pred_fft = torch.fft.fft(pred, dim=1)
    target_fft = torch.fft.fft(target, dim=1)

    pred_power = torch.abs(pred_fft) ** 2
    target_power = torch.abs(target_fft) ** 2

    pred_power = pred_power / (pred_power.sum(dim=1, keepdim=True) + 1e-8)
    target_power = target_power / (target_power.sum(dim=1, keepdim=True) + 1e-8)

    return F.mse_loss(pred_power, target_power)




def CorrelationLoss(pred, target, eps=1e-8):
    """
    Correlation-based loss function (1 - Pearson correlation).
    Minimizing this pushes predictions and targets to be highly correlated.
    """

    # Flatten if multi-dimensional
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    # Demean
    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)

    # Numerator: covariance
    cov = (pred_mean * target_mean).sum(dim=1)

    # Denominator: product of standard deviations
    pred_std = torch.sqrt((pred_mean**2).sum(dim=1) + eps)
    target_std = torch.sqrt((target_mean**2).sum(dim=1) + eps)

    corr = cov / (pred_std * target_std + eps)

    # Loss = 1 - correlation (maximize correlation → minimize loss)
    loss = 1 - corr.mean()
    return loss

def totalPrecipLoss(pred, target):
    """
    Loss function based on total precipitation difference.
    """
    total_pred = pred.sum(dim=1)  # Sum over time dimension
    total_target = target.sum(dim=1)

    loss = F.mse_loss(total_pred, total_target)
    return loss


def spatial_correlation_loss(yhat, ytrue, eps=1e-8):
    B,P,T = yhat.shape
    yh = yhat - yhat.mean(dim=1, keepdim=True)
    yt = ytrue - ytrue.mean(dim=1, keepdim=True)

    if yh.shape[2] != yt.shape[2]:
        # Resample to the same temporal resolution using nearest neighbor
        T_out = min(yh.shape[2], yt.shape[2])
        yh = resample_time_nearest(yh, T_out)
        yt = resample_time_nearest(yt, T_out)

    num = (yh * yt).sum(dim=1)            # (B,T)
    den = (yh.square().sum(dim=1).clamp_min(eps).sqrt() *
           yt.square().sum(dim=1).clamp_min(eps).sqrt()) 
    corr = num / den                      # no clamp
    return (1.0 - corr).mean()


def resample_time_nearest(x, T_out):
    B, P, T_in = x.shape
    y = x.reshape(B*P, 1, T_in)
    y = F.interpolate(y, size=T_out, mode='nearest')   # no smoothing
    return y.reshape(B, P, T_out)

