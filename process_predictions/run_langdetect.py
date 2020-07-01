import logging
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing
import joblib
import glob
import langdetect
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


DATA_DIR = os.path.join('..', 'data')

def detect_lang(text):
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return None

def langdetect_file(f_name, output_folder):
    df = pd.read_parquet(f_name, columns=['user.description'])
    df['userbio_lang'] = df['user.description'].apply(detect_lang)
    f_out_name = os.path.basename(f_name)
    f_out = os.path.join(output_folder, f_out_name)
    df[['userbio_lang']].to_parquet(f_out)

def main():
    # paths
    run_dir = os.path.join(DATA_DIR, 'prediction_data', 'run_2020_06_29-11-06_1593421617')
    parquet_files = glob.glob(os.path.join(run_dir, 'parquet', '*parquet'))
    output_path = os.path.join(run_dir, 'userbio_lang')
    # create dirs
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    # run langdetect
    num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cpus)
    langdetect_file_delayed = joblib.delayed(langdetect_file)
    parallel(langdetect_file_delayed(f_name, output_path) for f_name in tqdm(parquet_files))


if __name__ == "__main__":
    main()
