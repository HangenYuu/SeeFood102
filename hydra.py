import hydra

from omegaconf import OmegaConf

config = OmegaConf.load('config.yaml')
print(config.preferences.user)
print(config.preferences.trait)

@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg):
    # Print the config file using `to_yaml` method which prints in a pretty manner
    print(OmegaConf.to_yaml(cfg))
    print(cfg.preferences.user)

if __name__ == "__main__":
    main()