import argparse
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import OmegaConf
import l2winddir.simple_conv
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

def load_from_cfg(
    cfg_path,
    key,
    overrides=None,
    overrides_targets=None,
    cfg_hydra_path=None,
    call=True,
):
    """
    Load an object from a configuration file.

    Parameters
    ----------
    cfg_path : str or Path
        Path to the configuration file.
    key : str or None
        If not None, the value at the given key in the configuration will be loaded.
    overrides : dict, optional
        Additional configuration overrides.
    overrides_targets : dict, optional
        Additional configuration overrides for `_target_` fields.
    cfg_hydra_path : str or Path, optional
        Path to the Hydra configuration file.
    call : bool, optional
        Whether to call the loaded object.

    Returns
    -------
    obj : any
        The loaded object.
    """
    src_cfg = OmegaConf.load(Path(cfg_path))
    overrides = overrides or dict()
    OmegaConf.set_struct(src_cfg, True)
    if cfg_hydra_path is not None:
        hydra_cfg = OmegaConf.load(Path(cfg_hydra_path))
        OmegaConf.register_new_resolver(
            "hydra", lambda k: OmegaConf.select(hydra_cfg, k), replace=True
        )
    # with OmegaConf.open_dict(src_cfg):
    cfg = OmegaConf.merge(src_cfg, overrides)
    if overrides_targets is not None:
        for path, target in overrides_targets.items():
            node = OmegaConf.select(cfg, path)
            node._target_ = target
    if key is not None:
        cfg = OmegaConf.select(cfg, key)
    return hydra.utils.call(cfg) if call else cfg


def load_model(xp_dir, cfg_path, hydra_cfg_path):
    """
    Load a model and its checkpoint from specified directories.

    Parameters
    ----------
    xp_dir : Path
        The path to the experiment directory containing model checkpoints.
    cfg_path : str or Path
        Path to the configuration file for loading the model.
    hydra_cfg_path : str or Path
        Path to the Hydra configuration file.

    Returns
    -------
    model : torch.nn.Module
        The loaded model with weights restored from the checkpoint.
    ckpt : dict
        The checkpoint dictionary containing model state and other metadata.

    Raises
    ------
    FileNotFoundError
        If no checkpoint file is found in the specified directory.
    KeyError
        If the checkpoint does not contain the 'state_dict' key.
    """
    model = load_from_cfg(
        cfg_path=str(cfg_path),
        key="model",
        cfg_hydra_path=str(hydra_cfg_path),
    )

    ckpt_xp_dir = xp_dir / "wind_direction_model"
    ckpt_path = next(ckpt_xp_dir.glob("*/checkpoints/*.ckpt"), None)

    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_xp_dir / 'checkpoints/'}"
        )

    ckpt = torch.load(ckpt_path)
    if "state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'state_dict'")

    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def load_data_module(data_path, cfg_path, hydra_cfg_path, checkpoint):
    """
    Load a LightningDataModule and set it up for prediction.

    Parameters
    ----------
    data_path : str or Path
        Path to the data file to be used for prediction.
    cfg_path : str or Path
        Path to the configuration file for the data module.
    hydra_cfg_path : str or Path
        Path to the Hydra configuration file.
    checkpoint : dict
        The checkpoint dictionary containing the training mean and standard deviation.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        The test data loader set up for prediction.

    """
    if isinstance(data_path, str):
        data_module = load_from_cfg(
            cfg_path=str(cfg_path),
            key="data_module",
            cfg_hydra_path=str(hydra_cfg_path),
            overrides=dict(data_module=dict(test_data_paths=data_path)),
        )
    elif isinstance(data_path, xr.Dataset):
        data_module = load_from_cfg(
            cfg_path=str(cfg_path),
            key="data_module",
            cfg_hydra_path=str(hydra_cfg_path),
        )
        data_module.test_data_paths = data_path
    data_module.train_mean = checkpoint["train_mean"]
    data_module.train_std = checkpoint["train_std"]
    data_module.setup(stage="predict")
    dataloader = data_module.test_dataloader()
    return dataloader


def predict(model_path, data_path, model=None, checkpoint=None):
    """
    Generate predictions and uncertainties from a trained model using given data.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained model's directory.
    data_path : str or Path
        Path to the dataset for which predictions are to be made.
    model : torch.nn.Module, optional
        Pre-loaded model, if available. If None, the model will be loaded from the checkpoint.
    checkpoint : dict, optional
        Pre-loaded checkpoint dictionary containing model state and metadata. If None, it will be loaded.

    Returns
    -------
    pred_deg : numpy.ndarray
        Array of predicted wind directions in degrees.
    pred_uncert : numpy.ndarray
        Array of prediction uncertainties in degrees.
    """
    xp_dir = Path(model_path).parents[3]
    cfg_path = xp_dir / ".hydra/config.yaml"
    hydra_cfg_path = xp_dir / ".hydra/hydra.yaml"
    if model is None or checkpoint is None:
        model, checkpoint = load_model(xp_dir, cfg_path, hydra_cfg_path)
    dataloader = load_data_module(data_path, cfg_path, hydra_cfg_path, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    y_pred = []
    pred_uncertainty = []

    with torch.no_grad():
        for batch in dataloader:
            X, _ = batch
            X = X.to(device)

            result = model(X)
            if isinstance(result, tuple):
                preds, uncer = result
                y_pred.append(preds.cpu())
                pred_uncertainty.append(uncer.cpu())
            else:
                y_pred.append(result.cpu())
                pred_uncertainty.append(torch.empty_like(result).fill_(float("nan")))

    y_pred = torch.cat(y_pred, dim=0).squeeze()
    pred_uncertainty = torch.cat(pred_uncertainty, dim=0).squeeze()

    pred_deg = np.degrees(y_pred.numpy()) % 180
    pred_uncert = np.degrees(pred_uncertainty.numpy()) % 180

    return pred_deg, pred_uncert


def make_prediction(model_path, data_path, eval=False, save_path="."):
    """
    Make predictions and uncertainties from a trained model using given data.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained model's directory.
    data_path : str or Path
        Path to the dataset for which predictions are to be made.
    eval : bool
        If True, return a pandas DataFrame containing predictions and uncertainties.
        If False, return an xarray Dataset with the predictions and uncertainties added as new variables.
    save_path : str or Path
        Path to save the resulting dataset if eval=False.

    Returns
    -------
    If eval=True, a pandas DataFrame with predictions and uncertainties.
    If eval=False, an xarray Dataset with the predictions and uncertainties added as new variables.
    """
    model_path = Path(model_path)
    xp_dir = model_path.parents[3]
    cfg_path = xp_dir / ".hydra/config.yaml"
    hydra_cfg_path = xp_dir / ".hydra/hydra.yaml"
    model, checkpoint = load_model(xp_dir, cfg_path, hydra_cfg_path)

    model_name = Path(model_path).parts[-3]
    y_pred, uncertainty = predict(
        model=model,
        checkpoint=checkpoint,
        model_path=model_path,
        data_path=data_path,
    )
    if eval:
        df = pd.DataFrame(
            {"predicted winddir": y_pred, "predicted uncert": uncertainty}
        )
        return df
    else:
        if isinstance(data_path, str):
            ds = xr.open_dataset(data_path)
        elif isinstance(data_path, xr.Dataset):
            ds = data_path
        headin_angle = ds["ground_heading"][0][0][0].values
        y_pred_geo_ref = np.degrees(np.radians(y_pred) + np.radians(headin_angle)) % 180
        ds = ds.assign(
            predicted_wind_direction=("tile", y_pred_geo_ref),
            predicted_uncertainty=("tile", uncertainty),
        )
        ds["predicted_wind_direction"].attrs["units"] = "degrees"
        ds["predicted_wind_direction"].attrs["comment"] = (
            f"Geographical reference predicted wind direction done by model: {model_name}"
        )
        ds["predicted_uncertainty"].attrs["units"] = "degrees"
        ds["predicted_uncertainty"].attrs["comment"] = (
            f"Uncertainty of predicted wind direction done by model: {model_name}"
        )
        return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predict wind direction")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data",
    )
    parser.add_argument(
        "--eval",
        type=bool,
        default=False,
        help="Evaluate the model",
    )
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    eval = args.eval
    make_prediction(model_path=model_path, data_path=data_path, eval=eval)
