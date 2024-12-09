import torch


def compute_velocity_error(prediction, target, average=True):
    prediction_mag = torch.linalg.norm(prediction, dim=-1)
    target_mag = torch.linalg.norm(target, dim=-1)
    if average:
        return torch.mean(torch.abs(prediction_mag - target_mag))
    else:
        return torch.abs(prediction_mag - target_mag)


def compute_direction_error(prediction, target, average=True):
    prediction_dir = torch.atan2(prediction[..., 1], prediction[..., 0])
    target_dir = torch.atan2(target[..., 1], target[..., 0])
    # convert to deg
    if average:
        return (180 / torch.pi) * torch.mean(torch.abs(prediction_dir - target_dir))
    else:
        return torch.abs(prediction_dir - target_dir)

def compute_and_save_my_metrics(self, loss, prediction, prediction_gt, val=False):

    velocity_diff = compute_velocity_error(prediction, prediction_gt)
    direction_diff = compute_direction_error(prediction, prediction_gt)

    prefix = "val" if val else "train"

    self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
    self.log(
        f"{prefix}_vel_diff",
        velocity_diff,
        on_step=True,
        on_epoch=False,
        prog_bar=False,
    )
    self.log(
        f"{prefix}_dir_diff",
        direction_diff,
        on_step=True,
        on_epoch=False,
        prog_bar=False,
    )