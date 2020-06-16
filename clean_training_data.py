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

    # convert types labels
    logger.info('Sanitize label names...')
    for col in ['type1', 'type2', 'type3', 'majority_vote_type']:
        df.loc[:, col] = df[col].apply(lambda s: convert_types[s])

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


    train, test = sklearn.model_selection.train_test_split(df, test_size=.2, random_state=seed, shuffle=True)
    logger.info(f'Writing train/test data...')
    write_df(train, 'train')
    write_df(test, 'test')

def write_df(df, f_name):
    f_out = os.path.join(input_folder, f_name)
    df.to_csv(f_out + '.csv', index=False)
    df.to_pickle(f_out + '.pkl')

def preprocess(text):
    # demojize
    text = de_emojize(text)
    # standardize text
    text = standardize_text(text)
    # anonymize
    text = anonymize_text(text)
    return text

def standardize_text(text):
    """
    1) Escape HTML
    2) Replaces some non-standard punctuation with standard versions. 
    3) Replace \r, \n and \t with white spaces
    4) Removes all other control characters and the NULL byte
    5) Removes duplicate white spaces
    """
    # escape HTML symbols
    text = html.unescape(text)
    # standardize punctuation
    text = text.translate(transl_table)
    text = text.replace('…', '...')
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, ' ', text)
    # remove all remaining control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    return text.strip()

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

def anonymize_text(text, url_filler='<url>', user_filler='@user', email_filler='@email'):
    text = replace_urls(text, filler=url_filler)
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
