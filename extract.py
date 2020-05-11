import pandas as pd
import glob
from collections import defaultdict
import gzip
import json
import os
from utils.extract_tweet import ExtractTweet
from utils.geo_helpers import load_map_data, convert_to_polygon
from covid_19_keywords import KEYWORDS
from local_geocode.geocode.geocode import Geocode
import shapely
from tqdm import tqdm
import logging
import joblib
import multiprocessing

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = 'extracted/tweets'
OTHER_DIR = 'extracted/other'


def get_file_names_by_hour():
    """Group files by hour"""
    f_names = glob.glob('data/**/**/**/**/**')
    f_names_by_hour = defaultdict(list)
    for f_name in f_names:
        key = '_'.join(f_name.split('/')[1:5])
        f_names_by_hour[key].append(f_name)
    return f_names_by_hour

def get_matched_keywords(text):
    text = text.lower()
    return [i for i, keyword in enumerate(KEYWORDS) if keyword.lower() in text]

def get_geo_info(tweet, map_data, gc):
    """ 
    Tries to infer differen types of geoenrichment from tweet
    0) no geo enrichment could be done
    1) coordinates: Coordinates were provided in tweet object
    2) place_centroid: Centroid of place bounding box
    3) user location: Infer geo-location from user location

    Returns dictionary with the following keys:
    - longitude (float)
    - latitude (float)
    - country_code (str)
    - region (str)
    - subregion (str)
    - geo_type (int)

    Regions (according to World Bank):
    East Asia & Pacific, Latin America & Caribbean, Europe & Central Asia, South Asia,
    Middle East & North Africa, Sub-Saharan Africa, North America, Antarctica

    Subregions:
    South-Eastern Asia, South America, Western Asia, Southern Asia, Eastern Asia, Eastern Africa,
    Northern Africa Central America, Middle Africa, Eastern Europe, Southern Africa, Caribbean,
    Central Asia, Northern Europe, Western Europe, Southern Europe, Western Africa, Northern America,
    Melanesia, Antarctica, Australia and New Zealand, Polynesia, Seven seas (open ocean), Micronesia
    """
    def get_region_by_country_code(country_code):
        return map_data[map_data['ISO_A2'] == country_code].iloc[0].REGION_WB

    def get_subregion_by_country_code(country_code):
        return map_data[map_data['ISO_A2'] == country_code].iloc[0].SUBREGION

    def get_country_code_by_coords(longitude, latitude):
        coordinates = shapely.geometry.point.Point(longitude, latitude)
        within = map_data.geometry.apply(lambda p: coordinates.within(p))
        if sum(within) > 0:
            return map_data[within].iloc[0].ISO_A2
        else:
            logger.warning(f'Could not match country for coordinates {longitude}, {latitude}')
            return None

    geo_obj = {
            'longitude': None,
            'latitude': None,
            'country_code': None,
            'region': None,
            'subregion': None,
            'geo_type': 0
            }

    if tweet.has_coordinates:
        # try to get geo data from coordinates (<0.1% of tweets)
        geo_obj['longitude'] = tweet.tweet['coordinates']['coordinates'][0]
        geo_obj['latitude'] = tweet.tweet['coordinates']['coordinates'][1]
        geo_obj['country_code'] = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
        geo_obj['geo_type'] = 1
    elif tweet.has_place_bounding_box:
        # try to get geo data from place (roughly 1% of tweets)
        p = convert_to_polygon(tweet.tweet['place']['bounding_box']['coordinates'][0])
        geo_obj['longitude'] = p.centroid.x
        geo_obj['latitude'] = p.centroid.y
        country_code = tweet.tweet['place']['country_code']
        if country_code == '':
            # sometimes places don't contain country codes, try to resolve from coordinates
            country_code = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
        geo_obj['country_code'] = country_code
        geo_obj['geo_type'] = 2
    else:
        # try to parse user location
        locations = gc.decode(tweet.tweet['user']['location'])
        if len(locations) > 0:
            geo_obj['longitude'] = locations[0]['longitude']
            geo_obj['latitude'] = locations[0]['latitude']
            country_code = locations[0]['country_code']
            if country_code == '':
                # sometimes country code is missing (e.g. disputed areas), try to resolve from geodata
                country_code = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
            geo_obj['country_code'] = country_code
            geo_obj['geo_type'] = 3
    if geo_obj['country_code']:
        # retrieve region info
        if geo_obj['country_code'] in map_data.ISO_A2.tolist():
            geo_obj['region'] = get_region_by_country_code(geo_obj['country_code'])
            geo_obj['subregion'] = get_subregion_by_country_code(geo_obj['country_code'])
        else:
            logger.warning(f'Unknown country_code {geo_obj["country_code"]}')
    return geo_obj

def process_files_by_hour(key, f_names):
    """Process files from same hour"""
    tweet_interaction_counts = defaultdict(lambda: {'num_quotes': 0, 'num_retweets': 0, 'num_replies': 0})
    gc = Geocode()
    gc.init()
    map_data = load_map_data()
    f_out_other_path = os.path.join(OTHER_DIR, f'{key}.jsonl')
    f_out_path = os.path.join(OUTPUT_DIR, f'{key}.jsonl')
    for f_name in f_names:
        if f_name.endswith('.gz'):
            f = gzip.open(f_name, 'r')
        else:
            # older files were not gzipped
            f = open(f_name, 'r')
        for i, line in enumerate(f):
            tweet = json.loads(line)
            if not 'id' in tweet:
                # is not tweet, dump to differnt file for later inspection
                with open(f_out_other_path, 'a') as f_out:
                    f_out.write(json.dumps(tweet) + '\n')
                continue
            tweet = ExtractTweet(tweet)
            if tweet.is_retweet:
                tweet_interaction_counts[tweet.retweeted_status_id]['num_retweets'] += 1
                continue
            if tweet.has_quoted_status:
                tweet_interaction_counts[tweet.quoted_status_id]['num_quotes'] += 1
                continue
            if tweet.is_reply:
                tweet_interaction_counts[tweet.replied_status_id]['num_replies'] += 1
            extracted = tweet.extract_fields()
            extracted['matched_keywords'] = get_matched_keywords(tweet.text)
            geo_obj = get_geo_info(tweet, map_data, gc)
            extracted = {**extracted, **geo_obj}
            # write to file
            with open(f_out_path, 'a') as f_out:
                f_out.write(json.dumps(extracted) + '\n')
        f.close()
    return tweet_interaction_counts, f_out_path

def merge_interaction_counts(res):
    """Merges all interaction from different days to a final dict of tweet_id->num_replies, num_quotes, num_retweets"""
    tweet_interaction_counts = defaultdict(lambda: {'num_quotes': 0, 'num_retweets': 0, 'num_replies': 0})
    for r in tqdm(res):
        for k, v in r.items():
            for _type in v.keys():
                tweet_interaction_counts[k][_type] += v[_type]
    return tweet_interaction_counts

def write_parquet_file(output_file, interaction_counts):
    """Reads jsonl files, adds interaction counts and writes parquet files"""
    df = []
    with open(output_file, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweet = {**tweet, **interaction_counts[tweet['id']]}
            df.append(tweet)
    filename = os.path.basename(output_file).split('.jsonl')[0]
    f_out = os.path.join(OUTPUT_DIR, f'covid_stream_{filename}.parquet.gz')
    df = pd.DataFrame(df)
    df.to_parquet(f_out, compression='gzip') 
    # delete old file
    os.remove(output_file)
    return len(df)

def main():
    # set up geocode
    gc = Geocode()
    gc.prepare()

    # make sure map data is downloaded
    load_map_data()

    # set up parallel
    f_names_by_hour = get_file_names_by_hour()
    num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cpus)

    # extract fields and store write to intermediary jonsl files
    logger.info('Extract fields from tweets...')
    process_fn_delayed = joblib.delayed(process_files_by_hour)
    res = parallel((process_fn_delayed(key, f_names) for key, f_names in tqdm(f_names_by_hour.items())))

    # merge interaction data (num retweets, num quotes, num replies)
    logger.info('Merging all interaction counts...')
    interaction_counts = [dict(r[0]) for r in res]
    interaction_counts = merge_interaction_counts(interaction_counts)

    # add interaction data to tweets and write compressed parquet dataframes
    logger.info('Writing parquet files...')
    output_files = [r[1] for r in res]
    write_parquet_file_delayed = joblib.delayed(write_parquet_file)
    num_tweets = parallel((write_parquet_file_delayed(output_file, interaction_counts) for output_file in tqdm(output_files)))
    num_tweets = sum(s for s in num_tweets)
    logger.info(f'Wrote {len(output_files):,} files containing {num_tweets:,} tweets.... done!')

if __name__ == "__main__":
    main()
