import hydra
from omegaconf import DictConfig

from w2l.asr_train import train_asr


@hydra.main(config_name="config")
def main(config: DictConfig):
    train_asr(config)
