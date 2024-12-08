import torch

def compute_velocity_error(prediction, target,average=True):
    prediction_mag = torch.linalg.norm(prediction, dim=-1)
    target_mag = torch.linalg.norm(target, dim=-1)
    if average:
        return torch.mean(torch.abs(prediction_mag - target_mag))
    else:
        return torch.abs(prediction_mag - target_mag)

def compute_direction_error(prediction, target,average=True):
    prediction_dir = torch.atan2(prediction[..., 1], prediction[..., 0])
    target_dir = torch.atan2(target[..., 1], target[..., 0])
    if average:
        return torch.mean(torch.abs(prediction_dir - target_dir))
    else:
        return torch.abs(prediction_dir - target_dir)