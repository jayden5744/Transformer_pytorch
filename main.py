import hydra
from omegaconf import DictConfig

from src.tools import Trainer


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()



@hydra.main(config_path="config", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    pass


if __name__ == '__main__':
    train()
