import pandas as pd
import logging
from datetime import datetime
import glob
import os
import joblib
import multiprocessing


logger = logging.getLogger(__name__)


def get_parsed_data(usecols=None, s_date=None, e_date=None, read_in_parallel=True, num_files=None):
    """Read parsed data
    :param usecols: Only extract certain columns (default: all columns)
    :param s_date: start date to read from (str of format YYYY-MM-dd or datetime obj)
    :param e_date: end date to read from (str of format YYYY-MM-dd or datetime obj)
    :param read_in_parallel: read using multiple threads
    :param num_files: read first n parquet files
    """
    def parse_date_from_f_name(f_name):
        f_name = os.path.basename(f_name)
        date = f_name.split('.')[0][len('parsed_'):]
        if len(date.split('-')) == 3:
            # YYYY-MM-dd format
            return datetime.strptime(date, '%Y-%m-%d')
        else:
            # assume YYYY-MM-dd-HH format
            return datetime.strptime(date, '%Y-%m-%d-%H')

    def get_f_names(s_date=s_date, e_date=e_date):
        data_folder = get_data_folder()
        f_path = os.path.join(data_folder, '1_parsed', 'tweets', '*.parquet')
        f_names = glob.glob(f_path)
        if s_date is not None or e_date is not None:
            f_names_dates = {f_name: parse_date_from_f_name(f_name) for f_name in f_names}
            if s_date is not None:
                if not isinstance(s_date, datetime):
                    s_date = datetime.strptime(s_date, '%Y-%m-%d')
                f_names = [f_name for f_name in f_names if f_names_dates[f_name] > s_date]
            if e_date is not None:
                if not isinstance(e_date, datetime):
                    e_date = datetime.strptime(e_date, '%Y-%m-%d')
                f_names = [f_name for f_name in f_names if f_names_dates[f_name] < e_date]
        return f_names

    def read_data(f_name):
        """Reads single parquet file"""
        df = pd.read_parquet(f_name, columns=usecols)
        return df

    # load files
    f_names = get_f_names()
    if isinstance(num_files, int):
        f_names = f_names[:num_files]
    # set up parallel
    if read_in_parallel:
        n_jobs = max(multiprocessing.cpu_count() - 1, 1)
    else:
        n_jobs = 1 
    parallel = joblib.Parallel(n_jobs=n_jobs, prefer='threads')
    read_data_delayed = joblib.delayed(read_data)
    # load data
    logger.info('Reading data...')
    df = parallel(read_data_delayed(f_name) for f_name in tqdm(f_names))
    logger.info('Concatenating...')
    df = pd.concat(df)
    # convert to category
    for col in ['country_code', 'region', 'subregion', 'geo_type', 'lang']:
        if col in df: 
            df[col] = df[col].astype('category')
    return df
