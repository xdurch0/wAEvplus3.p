import hydra
from omegaconf import DictConfig

from w2l.conversion_train import train_conversion


@hydra.main(config_name="config")
def main(config: DictConfig):
    train_conversion(config)


if __name__ == "__main__":
    main()
