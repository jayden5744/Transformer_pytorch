import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    pass


@hydra.main(config_path="config", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    pass


if __name__ == '__main__':
    train()
