#import torch
from django.apps import AppConfig
#try:
#    # torch fork problem workaround
#    torch.set_num_threads(1)
#except RuntimeError:
#    pass


class DatabaseConfig(AppConfig):
    name = 'database'

