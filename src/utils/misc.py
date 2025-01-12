import torch
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler

def un_tan_fi(data):
    d = data.clone()
    d += 1
    d /= 2
    return d


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=(0, ),
                 restart_weights=(1, ),
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i

def get_metrics(img1, img2, p, z, Standardize=True):
    # if Standardize == True: 
    #     device = img1.device
    #     mean = torch.tensor([0.2400, 0.2518, 0.3112], device=device).view(3, 1, 1)
    #     std = torch.tensor([0.0892, 0.0785, 0.1011], device=device).view(3, 1, 1)

    #     img1 = img1 * std + mean
    #     img2 = img2 * std + mean
    
    #img1 is gt
    img1 = un_tan_fi(img1)
    img2 = un_tan_fi(img2)
    img1 = (img1 * 255.0).round().to(torch.uint8)
    img2 = (img2 * 255.0).round().to(torch.uint8)

    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    psnr_value = p(img2, img1)
    ssim_value = z(img2, img1)

    return psnr_value.item(), ssim_value.item()