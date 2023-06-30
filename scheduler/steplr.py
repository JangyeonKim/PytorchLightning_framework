import torch

def Scheduler(optimizer, step_size, gamma, **kwargs) :
    sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return sche_fn