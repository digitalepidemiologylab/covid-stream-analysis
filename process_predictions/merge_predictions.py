""""
Prediction files are generated as jsonl files and are located in ../data/prediction_data/<run_name>/predictions/<prediction_run_name>/predictions/*.jsonl

Predicted lang of user bio files (see run_langdetect.py) are located in ../data/prediction_data/<run_name>/userbio_lang

Raw data is in ../data/prediction_data/<run_name>/(parquet/txt)
"""

import os
import sys
import glob
import multiprocessing
import joblib
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join('..', 'data')
run_name = 'run_2020-07-17_17-37-18_734268_multilang'
category_run_name = 'predictions_2020-07-17_16-18-23_341756'
type_run_name = 'predictions_2020-07-17_16-18-29_216072'

def process(f_name, file_paths, output_dir):
    # read all data
    df_lang = pd.read_parquet(file_paths['lang_detect_path'])
    df_category = pd.read_json(file_paths['category_path'], lines=True)
    df_category = df_category.rename(columns={'label': 'category_label', 'label_probabilities': 'category_probabilities'})
    df_type = pd.read_json(file_paths['type_path'], lines=True)
    df_type = df_type.rename(columns={'label': 'type_label', 'label_probabilities': 'type_probabilities'})
    df = pd.read_parquet(file_paths['raw_path'], columns=['user.id'])

    # get rid of index
    df_lang = df_lang.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, df_type, df_category, df_lang], axis=1)

    # convert to category
    for col in ['type_label', 'category_label', 'userbio_lang']:
        df[col] = df[col].astype('category')

    # write merged df
    df.to_parquet(os.path.join(output_dir, f'{f_name}.parquet'))


def main():
    # paths
    run_dir = os.path.join(DATA_DIR, 'prediction_data', run_name)
    output_dir = os.path.join(run_dir, 'merged_predictions')
    lang_detect_paths = glob.glob(os.path.join(run_dir, 'userbio_lang', '*.parquet'))
    category_paths = glob.glob(os.path.join(run_dir, 'predictions', category_run_name, 'predictions', '*.jsonl'))
    type_paths = glob.glob(os.path.join(run_dir, 'predictions', type_run_name, 'predictions', '*.jsonl'))
    raw_paths = glob.glob(os.path.join(run_dir, 'parquet', '*.parquet'))

    # dirs
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # compile paths
    f_names_dict = {}
    for lang_detect_path, category_path, type_path, raw_path in zip(sorted(lang_detect_paths), sorted(category_paths), sorted(type_paths), sorted(raw_paths)):
        f_name = os.path.basename(lang_detect_path).split('.parquet')[0]
        f_names_dict[f_name] = {
                'lang_detect_path': lang_detect_path,
                'category_path': category_path,
                'type_path': type_path,
                'raw_path': raw_path
                }

    # set up parallel
    num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cpus)
    process_delayed = joblib.delayed(process)

    # run
    parallel(process_delayed(f_name, f_paths, output_dir) for f_name, f_paths in tqdm(f_names_dict.items()))


if __name__ == "__main__":
    main()
