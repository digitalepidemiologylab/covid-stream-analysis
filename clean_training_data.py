import pandas as pd
import html
import unicodedata
import re
from bs4 import BeautifulSoup
import ast
import logging
import os
import warnings
import sklearn.model_selection
import unidecode
import shutil

# mute beautiful soup warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# compile regexes
control_char_regex = r'[\r\n\t]+'
transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”–-",  u"'''\"\"--")])
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
email_regex = re.compile(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+')

convert_types = {
        'Individual:Male': 'individual_male',
        'Institution': 'institution',
        'Individual: Other gender or unclear': 'individual_unclear',
        'Unclear': 'unclear',
        'Individual:Female': 'individual_female',
        '': None
        }

convert_types_merged = {
        'Individual:Male': 'individual',
        'Institution': 'institution',
        'Individual: Other gender or unclear': 'individual',
        'Unclear': 'unclear',
        'Individual:Female': 'individual',
        '': None
        }

convert_categories = {
        'Other': 'other',
        'Media: News': 'media_news',
        'Media: Other Media': 'media_other',
        'Government and Politics': 'politics',
        'Outspoken Political Supporter': 'political_supporter',
        'Media: Scientific News and Communication': 'media_scientific',
        'Art': 'art',
        'Business': 'business',
        'Non-Governmental Organization': 'ngo',
        'Healthcare': 'healthcare',
        'Science: Social Sciences': 'science_social',
        'Public Services': 'public_services',
        'Science: Engineering and Technology': 'science_engineering',
        'Sport': 'sports',
        'Science: Other Science': 'science_other',
        'Religion': 'religion',
        'Science: Life Sciences': 'science_lifescience',
        'Porn': 'porn',
        'Not in English': 'non_english',
        '': None
        }

convert_merged_categories = {
        'Other': 'other',
        'Media: News': 'media',
        'Media: Other Media': 'media',
        'Government and Politics': 'politics',
        'Outspoken Political Supporter': 'political_supporter',
        'Media: Scientific News and Communication': 'media',
        'Art': 'art',
        'Business': 'business',
        'Non-Governmental Organization': 'ngo',
        'Healthcare': 'healthcare',
        'Science: Social Sciences': 'science',
        'Public Services': 'public_services',
        'Science: Engineering and Technology': 'science',
        'Sport': 'sports',
        'Science: Other Science': 'science',
        'Religion': 'religion',
        'Science: Life Sciences': 'science',
        'Porn': 'porn',
        'Not in English': 'non_english',
        '': None
        }

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# globals
input_folder = os.path.join('data', 'annotation_data')

# Data overview
# types
# Individual:Male                        4458
# Institution                            3208
# Individual: Other gender or unclear    2485
# Unclear                                1970
# Individual:Female                      1955

# categories
# Other                                       6258
# Media: News                                  990
# Media: Other Media                           909
# Government and Politics                      782
# Outspoken Political Supporter                653
# Media: Scientific News and Communication     651
# Art                                          621
# Business                                     536
# Non-Governmental Organization                449
# Healthcare                                   420
# Science: Social Sciences                     331
# Public Services                              323
# Science: Engineering and Technology          311
# Sport                                        251
# Science: Other Science                       181
# Religion                                     169
# Science: Life Sciences                       128
# Porn                                          88
# Not in English                                25

def main(seed=42):
    f_path = os.path.join(input_folder, 'annotated_users_ten_languages-2.pickle')
    df = pd.read_pickle(f_path)

    # Clean bio
    logger.info('Clean bio text...')
    df = df[df.bio.apply(lambda s: isinstance(s, str))]
    df.loc[:, 'text'] = df.bio.apply(preprocess)
    df = df.drop_duplicates(subset=['text'])

    # convert types labels and add merged type
    logger.info('Sanitize label names...')
    for col in ['type1', 'type2', 'type3', 'majority_vote_type']:
        df_col = df[col].copy()
        df.loc[:, col] = df_col.apply(lambda s: convert_types[s])
        df[col + '_merged'] = df_col.apply(lambda s: convert_types_merged[s])

    # convert category labels and add merged category
    for col in ['category1', 'category2', 'category3', 'majority_vote_category']:
        df_col = df[col].copy()
        if col == 'majority_vote_category':
            df.loc[:, col] = df_col.apply(lambda s: convert_categories[s])
            df[col + '_merged'] = df_col.apply(lambda s: convert_merged_categories[s])
        else:
            df.loc[:, col] = df_col.apply(lambda s: [convert_categories[_s] for _s in s])
            df[col + '_merged'] = df_col.apply(lambda s: list(set([convert_merged_categories[_s] for _s in s])))

    # clear nan labels
    logger.info('Clean labels...')
    df = df.dropna(subset=['majority_vote_category', 'majority_vote_type'])

    # find unambiguous for type and category
    df['type_is_ambiguous'] = is_ambiguous(df, 'type1', 'type2', 'type3')
    df['type_merged_is_ambiguous'] = is_ambiguous(df, 'type1_merged', 'type2_merged', 'type3_merged')
    df['category_is_ambiguous'] = is_ambiguous(df, 'category1', 'category2', 'category3', multi_annotation=True)
    df['category_merged_is_ambiguous'] = is_ambiguous(df, 'category1_merged', 'category2_merged', 'category3_merged', multi_annotation=True)

    train, test = sklearn.model_selection.train_test_split(df, test_size=.2, random_state=seed, shuffle=True)
    logger.info(f'Writing train/test data...')
    write_df(train, 'train', seed)
    write_df(test, 'test', seed)

def is_ambiguous(df, col1, col2, col3, multi_annotation=False):
    if multi_annotation:
        # all annotations need to be equal and have only a single annotation
        return df.apply(lambda row: not (len(row[col1]) == len(row[col2]) == len(row[col3]) == 1) or not (row[col1][0] == row[col2][0] == row[col3][0]), axis=1)
    else:
        # all annotations need to be equal
        return df.apply(lambda row: not (row[col1] == row[col2] == row[col3]), axis=1)

def write_df(df, dataset, seed):
    f_out_folder = os.path.join(input_folder, dataset)
    # wipe previous data
    if os.path.isdir(f_out_folder):
        shutil.rmtree(f_out_folder)
    os.makedirs(f_out_folder)
    dfs = {}
    # full datasets
    dfs['type'] = df.rename(columns={'majority_vote_type': 'label'})[['text', 'label']].copy()
    dfs['type_merged'] = df.rename(columns={'majority_vote_type_merged': 'label'})[['text', 'label']].copy()
    dfs['category'] = df.rename(columns={'majority_vote_category': 'label'})[['text', 'label']].copy()
    dfs['category_merged'] = df.rename(columns={'majority_vote_category_merged': 'label'})[['text', 'label']].copy()
    # unambiguous datasets
    dfs['type_unambiguous'] = df[~df['type_is_ambiguous']].rename(columns={'majority_vote_type': 'label'})[['text', 'label']].copy()
    dfs['type_merged_unambiguous'] = df[~df['type_merged_is_ambiguous']].rename(columns={'majority_vote_type_merged': 'label'})[['text', 'label']].copy()
    dfs['category_unambiguous'] = df[~df['category_is_ambiguous']].rename(columns={'majority_vote_category': 'label'})[['text', 'label']].copy()
    dfs['category_merged_unambiguous'] = df[~df['category_merged_is_ambiguous']].rename(columns={'majority_vote_category_merged': 'label'})[['text', 'label']].copy()
    for name, df in dfs.items():
        f_out_folder_name = os.path.join(f_out_folder, name)
        if not os.path.isdir(f_out_folder_name):
            os.makedirs(f_out_folder_name)
        f_out = os.path.join(f_out_folder_name, f'all.csv')
        logger.info(f'Writing {len(df):,} examples to file {f_out}...')
        df.to_csv(f_out, index=False)
        # additionally do train/dev split
        if dataset == 'train':
            train, dev = sklearn.model_selection.train_test_split(df, test_size=.2, random_state=seed, shuffle=True)
            for _type, df_type in zip(['train', 'dev'], [train, dev]):
                f_out = os.path.join(f_out_folder_name, f'{_type}.csv')
                logger.info(f'Writing {len(df_type):,} examples to file {f_out}...')
                df_type.to_csv(f_out, index=False)

def preprocess(text):
    # demojize
    text = de_emojize(text)
    # separate hashtags
    text = separate_hashtags(text)
    # standardize text
    text = standardize_text(text)
    # anonymize
    text = anonymize_text(text)
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    text = text.strip()
    return text

def standardize_text(text):
    # escape HTML symbols
    text = html.unescape(text)
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, ' ', text)
    # remove all remaining control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # standardize all ligature/letter/punctuation characters
    text = ''.join([unidecode.unidecode(t) if unicodedata.category(t)[0] in 'LP' else t for t in text])
    return text

def de_emojize(text):
    soup = BeautifulSoup(text, 'html.parser')
    spans = soup.find_all('span')
    if len(spans) == 0:
        return text
    while soup.span is not None:
        emoji_bytes = ast.literal_eval(soup.span.attrs['data-emoji-bytes'])
        emoji = bytes(emoji_bytes).decode()
        soup.span.replace_with(emoji)
    return soup.text

def separate_hashtags(text):
    text = re.sub(r"#(\w+)#(\w+)", r" #\1 #\2 ", text)
    return text

def anonymize_text(text, url_filler='<url>', user_filler='@user', email_filler='@email'):
    # remove wrong @ and #
    text = replace_urls(text, filler=url_filler)
    text = re.sub(r"@\ ", r"@", text)
    text = re.sub(r"#\ ", r"#", text)
    text = replace_usernames(text, filler=user_filler)
    text = replace_email(text, filler=email_filler)
    return text

def replace_email(text, filler='@email'):
    # replace other user handles by filler
    text = re.sub(email_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text

def replace_usernames(text, filler='@user'):
    # replace other user handles by filler
    text = re.sub(username_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text

def replace_urls(text, filler='<url>'):
    # replace other urls by filler
    text = re.sub(url_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text

if __name__ == "__main__":
    main()
