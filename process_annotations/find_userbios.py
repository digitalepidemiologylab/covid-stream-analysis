"""
Script to merge AMT data (containing original bios) with annotated dataset
"""

import pandas as pd
import logging
import os
import langdetect

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

DATA_DIR = os.path.join('..', 'data', 'annotation_data')

def detect_lang(text):
    if not isinstance(text, str):
        return None
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return None

def main():
    logger.info('Reading AMT data')
    original_data_dump = 'AMT_input_merged_emojis.csv'
    df_org = pd.read_csv(os.path.join(DATA_DIR, original_data_dump))

    logger.info('Reading v2 annotation data')
    f_path = os.path.join(DATA_DIR, 'V2_annotated_users.pickle')
    df = pd.read_pickle(f_path)
    __import__('pdb').set_trace()

    _df = pd.DataFrame()
    for name in [ 'bio', 'username', 'screenname', 'n_retweets', 'covid_tweeting_language']:
        _df[name] = pd.concat([df_org[f'{name}_{i}'] for i in range(1, 11)], axis=0)
    _df = _df.rename(columns={'bio': 'bio_original'})
    _df = _df.reset_index(drop=True)
    logger.info('Merge')
    df = df.merge(_df[['bio_original', 'username']], on='username', how='left')
    logger.info('Fill na values')
    df['bio_original'] = df.bio_original.fillna(df['bio'])
    logger.info('Apply lang detect')
    df['bio_original_lang'] = df['bio_original'].apply(detect_lang)
    f_out = os.path.join(DATA_DIR, 'V2_annotated_users_including_original.pickle')
    logger.info(f'Write output pickle to {f_out}')
    df.to_pickle(f_out)

if __name__ == "__main__":
    main()
