#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import grdtiler
from omegaconf import OmegaConf
from l2winddir import make_prediction

from _version import __version__


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def pred(input, config='./config.yaml', output="."):
    """
    Predict wind direction from a Sentinel-1 SAFE directory.

    Parameters
    ----------
    input : str
        Path to the Sentinel-1 SAFE directory.
    config : str, optional
        Path to the configuration file in YAML format. Defaults to './config.yaml'.
    output : str, optional
        Path to the output directory. Defaults to ".".

    Returns
    -------
    xarray.Dataset
        Predicted wind direction dataset.
    """
    try:
        # Load configuration
        conf = OmegaConf.load(config)
        logging.info(f"Configuration loaded successfully from {config}")

        # # Process input
        # safe_path = process_input(input)
        # logging.info(f"Input processed successfully: {safe_path}")

        # Tile generation
        _, tiles = grdtiler.tiling_prod(
            path=input,
            tile_size=conf["tile_size"],
            resolution=conf["resolution"],
            noverlap=conf["noverlap"],
            config_file=conf["grdtiler_config"],
        )
        logging.info(f"Tiles generated: {len(tiles.tile.values)} tiles")

        # Filter NaN tiles
        mask = tiles.land_mask.any(dim=("tile_line", "tile_sample")).values
        valid_tiles = tiles.tile.values[~mask]
        if len(valid_tiles) == 0:
            logging.warning("No valid tiles found after filtering. Returning empty dataset.")
            return None

        logging.info(f"Valid tiles retained: {len(valid_tiles)}")

        # Make prediction
        ds = make_prediction(
            model_path=conf["model_path"],
            data_path=tiles.sel(tile=valid_tiles),
            eval=False,
        )
        logging.info("Prediction completed successfully.")

        # Add necessary attributes to the dataset
        for attr in ['main_footprint', 'specialHandlingRequired']:
            if attr in tiles.attrs:
                ds.attrs[attr] = str(tiles.attrs[attr])

        # Format and preprocess the dataset
        ds = format_output(ds)
        # Add l2winddir_version to the dataset
        ds.attrs["l2winddir_version"] = __version__
        # Uncomment the following line to save the dataset
        filename = Path(input).name
        save_ds(ds, filename=filename, output=output)

        return ds

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise RuntimeError(f"Prediction failed due to: {e}")


def process_input(input_identifier):
    """
    Process the input identifier to resolve the SAFE path.

    Parameters
    ----------
    input_identifier : str
        Input identifier, such as the SAFE name.

    Returns
    -------
    str
        Resolved path to the SAFE directory.

    Raises
    ------
    ValueError
        If the satellite case is unhandled.
    """
    try:
        safe_name = input_identifier.split(" ")[0]
        if safe_name.startswith("S1"):
            safe = get_s1_path(safe_name)
        elif safe_name.startswith("RCM"):
            safe = get_rcm_path(safe_name)
        elif safe_name.startswith("RS2"):
            safe = get_rs2_path(safe_name)
        else:
            raise ValueError(f"Unhandled satellite case: {safe_name}. "
                             f"Supported cases are Sentinel-1 (S1), Radarsat-2 (RS2), and RCM.")
        logging.info(f"Resolved SAFE path for {safe_name}: {safe}")
        return safe
    except Exception as e:
        logging.error(f"Error processing input identifier {input_identifier}: {e}")
        raise

def format_output(ds):
    """
    Format and preprocess the dataset by computing mean values and dropping unnecessary variables.

    This function computes mean values for specific data variables like ground heading,
    incidence angle, NRCS, NRCS detrend, and NESZ for each polarization and tile if they exist
    in the dataset. It assigns these mean values as new data variables in the dataset with
    appropriate comments. Additionally, it drops unnecessary data variables, dimensions, and
    coordinates to streamline the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing data variables like ground heading, incidence angle, etc.

    Returns
    -------
    xarray.Dataset

        Formatted dataset with computed mean values and unnecessary variables removed.
    """
    
    # Compute and assign mean ground heading angle values for each polarization and tile if exists
    if 'ground_heading' in ds.data_vars:
        heading_angle_mean_data = ds.ground_heading.mean(dim=('tile_line', 'tile_sample'))
        ds = ds.assign(
            heading_angle_mean=heading_angle_mean_data.assign_attrs({
                'comment': 'Mean ground heading angle values for each polarization and tile.'
            })
        )

    # Compute and assign mean incidence angle values for each polarization and tile if exists
    if 'incidence' in ds.data_vars:
        incidence_mean_data = ds.incidence.mean(dim=('tile_line', 'tile_sample'))
        ds = ds.assign(
            incidence_mean=incidence_mean_data.assign_attrs({
                'comment': 'Mean incidence angle values for each polarization and tile.'
            })
        )
    if 'sigma0' in ds.data_vars:
        sigma0_mean_data = ds.sigma0.sel(pol=slice(None)).mean(dim=('tile_line', 'tile_sample'))
        ds = ds.assign(
            nrcs_mean=sigma0_mean_data.assign_attrs({
                'comment': 'Mean NRCS values for each polarization and tile.'
            })
        )

    # Compute and assign mean NRCS detrend values for each polarization and tile if exists
    if 'sigma0_detrend' in ds.data_vars:
        sigma0_detrend_mean_data = ds.sigma0_detrend.sel(pol=slice(None)).mean(dim=('tile_line', 'tile_sample'))
        ds = ds.assign(
            nrcs_detrend_mean=sigma0_detrend_mean_data.assign_attrs({
                'comment': 'Mean NRCS detrend values for each polarization and tile.'
            })
        )

    # Compute and assign mean NESZ values for each polarization and tile if exists
    if 'nesz' in ds.data_vars:
        nesz_mean_data = ds.nesz.sel(pol=slice(None)).mean(dim=('tile_line', 'tile_sample'))
        ds = ds.assign(
            nesz_mean=nesz_mean_data.assign_attrs({
                'comment': 'Mean NESZ values for each polarization and tile.'
            })
        )

    variables_to_drop = [
        'sigma0', 
        'sigma0_detrend', 
        'nesz', 
        'land_mask', 
        'longitude', 
        'latitude', 
        'ground_heading',
        'incidence',
        'tile_footprint'
    ]
    # Drop unnecessary variables if they exist
    for var in variables_to_drop:
        if var in ds.data_vars:
            ds = ds.drop_vars(var)
    dims_to_drop = ['tile_line', 'tile_sampe']
    for dim in dims_to_drop:
        if dim in ds.dims:
            ds = ds.drop_dims(dim)
    coords_to_drop = ['line', 'sample', 'spatial_ref']
    for coord in coords_to_drop:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)

    return ds
    
    

def save_ds(ds, filename, output):
    """
    Saves an xarray dataset to a NetCDF file with a structured filename based on input parameters.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    filename : str
        The base filename used to construct the output filename. It is expected to follow a specific naming convention
        that indicates the satellite and other metadata.
    output : str
        The directory path where the NetCDF file will be saved.

    Notes
    -----
    - The function generates a filename based on the naming conventions for different satellite missions (e.g., S1, RCM, RS2).
    - The polarization mode is derived from the filename and is included in the output filename.
    - The dataset is saved in the specified output directory with a '.nc' extension.
    - If the output directory does not exist, it is created.
    """

    prov_name = (filename.lower()).replace('grdm', 'wdr').replace('grdh', 'wdr').replace('grd', 'wdr').replace('sgf', 'wdr').split("_")
    
    if filename.startswith("S1"):
        name = "-".join(prov_name[:3] + [prov_name[3][-2:]] + prov_name[4:6] + prov_name[6:-1])

    elif filename.startswith("RCM") or filename.startswith("RS2"):
        if 'vv' in prov_name:
            if 'vh' in prov_name:
                pol = 'dv' 
            else:
                pol = 'sv'  
        elif 'hh' in prov_name:
            if 'hv' in prov_name:
                pol = 'dh'  
            else:
                pol = 'sv'  
        else:
            pol = 'unknown' 

        dt_obj = datetime.strptime(ds.start_date, '%Y-%m-%d %H:%M:%S.%f')
        start_date = dt_obj.strftime('%Y%m%dT%H%M%S').lower()
        dt_obj = datetime.strptime(ds.stop_date, '%Y-%m-%d %H:%M:%S.%f')
        stop_date = dt_obj.strftime('%Y%m%dT%H%M%S').lower()
        name = "-".join([prov_name[0]] + [prov_name[4]] + [prov_name[-1]] + [pol] + [start_date] + [stop_date] + ["_____"] + ["_____"])
    
    save_filename = (f"{output}/{name}.nc")
    Path(save_filename).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(save_filename)


def get_rcm_path(rcm_SAFE):
    """
    Generates the file path for a RADARSAT Constellation Mission (RCM) SAFE file based on its name and date.

    Args:
        rcm_SAFE (str): Name of the RCM SAFE file.

    Returns:
        str: Full file path for the specified RCM SAFE file.
    """
    base_path = '/home/datawork-cersat-public/provider/csa/satellite/l1/rcm'
    rcm_name = rcm_SAFE.split('_')
    rcm_m = rcm_name[0].lower().replace('rcm', 'rcm-')
    rcm_s = rcm_name[4].lower()
    rcm_date = datetime.strptime(''.join(rcm_name[5:7]), '%Y%m%d%H%M%S')
    rcm_year = rcm_date.date().year
    rcm_day = rcm_date.timetuple().tm_yday
    path = f"{base_path}/{rcm_m}/{rcm_s}/{rcm_year}/{rcm_day:03d}/{rcm_SAFE}"
    # if os.path.exists(path):
    #     print(f'\'{path}\' exist on archive')
    # else:
    #     print(f'{rcm_SAFE} do not exist on archive')
    return path


def get_rs2_path(rs2_SAFE):
    """
    Generates the file path for a RADARSAT-2 (RS2) SAFE file based on its name and date.

    Args:
        rs2_SAFE (str): Name of the RS2 SAFE file.

    Returns:
        str: Full file path for the specified RS2 SAFE file.
    """
    base_path = '/home/datawork-cersat-public/cache/project/sarwing/data/RS2/L1'
    name_split = rs2_SAFE.split('_')
    if 'HH' in name_split and 'HV' in name_split:
        polarization = 'HH_HV'
    elif 'HH' in name_split and not 'HV' in name_split:
        polarization = 'HH'
    elif ('VV' in name_split and 'VH' in name_split) or 'VVVH' in name_split:
        polarization = 'VV_VH'
    elif 'VV' in name_split and not 'VH' in name_split:
        polarization = 'VV'

        
    if not name_split[2][0].isdigit():                  
        date = datetime.strptime(name_split[5], '%Y%m%d')
        path = f"{base_path}/{polarization}/{date.year}/{date.timetuple().tm_yday:03d}/{rs2_SAFE}"
    else:
        date = datetime.strptime(name_split[1], '%Y%m%d')
        path = f"{base_path}/{polarization}/{date.year}/{date.timetuple().tm_yday:03d}/{rs2_SAFE}"
    
    return path 


def get_s1_path(safe, archive_dir='datarmor_mpc'):
    
    if '.SAFE' not in safe:
        safe += '.SAFE'
        
    if safe.startswith('S1'):
        year = safe[17:21]
        doy = datetime.strptime(safe[17:25], '%Y%m%d').strftime('%j')
        satellite = safe[:3]
        acquisition = safe.split('_')[1][:2]
        level = safe[12:13]
        
        satdir = 'sentinel-1a' if satellite == 'S1A' else 'sentinel-1b'
        subproddir = f'L{level}'
        base_path = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa'
        full_path = os.path.join(base_path, satdir, subproddir, acquisition, f"{safe[:14]}", year, doy, safe)
    
    else:
        raise ValueError("Unhandled satellite case. Supported cases are Sentinel-1 (S1) and Sentinel-3 (S3).")
    
    return full_path

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-v", "--version", action="store_true", help="Print version"
    )
    pre_args, remaining_args = pre_parser.parse_known_args()
    
    if pre_args.version:
        print(__version__)
        sys.exit()

    parser = argparse.ArgumentParser("Predict wind direction")
    parser.add_argument('--version', action='store_true', help="Display version information")
    parser.add_argument(
        "--config",
        type=str,
        default='./config.yaml',
        help="Path to the config file",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the SAFE directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default='./',
        help="Path to the output directory",
    )
    args = parser.parse_args()
    if args.version:
        logging.info("pred version 1.0.0")

    pred(input=args.input, config=args.config, output=args.output)

if __name__ == "__main__":
    main()