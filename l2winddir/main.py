import torch
import torch.nn as nn
import hydra_zen
from hydra_zen import builds, launch, make_config
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning as L
from omegaconf import OmegaConf
from l2winddir.data_module import WindDirectionDataModule
import simple_conv


def main_hydra(model, data_module, trainer):
    """
    Main function to fit, validate, and test the model using the given trainer and data module.

    Args:
        model (L.LightningModule): The model to train and evaluate.
        data_module (L.LightningDataModule): The data module providing data for training, validation, and testing.
        trainer (L.Trainer): The Lightning Trainer used for training and evaluation.

    Returns:
        dict: A dictionary containing the local variables `model`, `data_module`, `trainer`, and outputs from fitting, validation, and testing.
    """
    fit_output = trainer.fit(model, datamodule=data_module)
    val_output = trainer.validate(model, datamodule=data_module)
    test_output = trainer.test(model, datamodule=data_module)

    model.set_normalization_params(data_module.train_mean, data_module.train_std)

    best_model_path = trainer.checkpoint_callback.best_model_path
    torch.save({
        'state_dict': model.state_dict(),
        'train_mean': data_module.train_mean,
        'train_std': data_module.train_std
    }, best_model_path)

    print(f"Best model saved at {best_model_path}")
    return locals()


# Define the configuration for the model using Hydra-Zen
MyModelMDNConfig = builds(
    simple_conv.MyModel,
    inc=1,
    hid=128,
    depth=7,
    down=4,
    model_mdn=True,
    num_gaussians=2,
    weight_decay=None,
    lr=1e-3,
    drop=False,
    # data_augmentation=False,
)

# Define paths to data files (Renaud)
base_path = '/raid/localscratch/jrmiadan/analyse/cnn_project/pythonProject1'
train_data_paths = f'{base_path}/dataset_180/train_dataset_norm.nc'
valid_data_paths = f'{base_path}/dataset_180/validation_dataset_norm.nc'
test_data_paths = f'{base_path}/dataset_180/test_dataset_norm.nc'

# Define paths to data files (Robin)
# train_data_paths = f'{base_path}/dataset/train_dataset_robin.nc'
# valid_data_paths = f'{base_path}/dataset/validation_dataset_robin.nc'
# test_data_paths = f'{base_path}/dataset/validation_dataset_robin.nc'

# Define the configuration for the data module using Hydra-Zen
DMConfig = builds(
    WindDirectionDataModule,
    train_data_paths=train_data_paths,
    valid_data_paths=valid_data_paths,
    test_data_paths=test_data_paths,
    inc=1
)

# Define the logger for TensorBoard using Hydra-Zen
logger = builds(
    TensorBoardLogger,
    save_dir="${hydra:runtime.output_dir}",
    name="wind_direction_model",
    version=''
)

# Define the checkpoint callback for model saving using Hydra-Zen
checkpoint_callback = builds(
    ModelCheckpoint,
    monitor='val_loss',
    save_top_k=1,
    mode='min',
    filename='best-checkpoint'
)

# Define the early stopping callback using Hydra-Zen
early_stopping_callback = builds(
    EarlyStopping,
    monitor='val_loss',
    patience=15,
    mode='min'
)

# Define the trainer configuration using Hydra-Zen
TrainerConfig = builds(
    L.Trainer,
    max_epochs=100,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback
    ],
    accelerator='gpu',
    devices=[7],
    precision=32,
    logger=logger,
    gradient_clip_val=0.05,
    gradient_clip_algorithm="value",
    log_every_n_steps=2
)

# Create a configuration object
config = make_config(
    trainer=TrainerConfig,
    data_module=DMConfig,
    model=MyModelMDNConfig
)

# Define overrides for the configuration
overrides = [
    "trainer.max_epochs=100",
    "data_module.inc=2",
    "model.inc=${data_module.inc}",
    "model.down=4",
    "model.hid=128",
    "model.depth=7",
    "model.model_mdn=True",
    "model.num_gaussians=2",
    "model.weight_decay=1e-4",
    "model.lr=1e-3",
    "model.drop=True",
    # "model.data_augmentation=True",
    "trainer.logger.version=${model.inc}_${model.down}_${model.hid}_${model.depth}_${model.model_mdn}_${model.weight_decay}_${model.drop}_ren_ds"
]

# Launch the Hydra-Zen jobs with the specified configuration and overrides
jobs = launch(
    config,
    hydra_zen.zen(main_hydra),
    overrides=overrides,
    version_base="1.3",
    multirun=True,
)

# Log the hyperparameters and metrics for each job
for job in jobs[0]:
    logger = job._return_value['trainer'].logger
    metrics = job._return_value['val_output'][0] | job._return_value['test_output'][0]
    logger.log_hyperparams(params=OmegaConf.to_container(job.cfg), metrics=metrics)
