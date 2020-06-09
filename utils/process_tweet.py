import pandas as pd
from copy import copy
from collections import defaultdict
import spacy
import hashlib
import shapely.geometry
import pickle
import itertools
from html.parser import HTMLParser
import unicodedata
import re
import logging
from functools import lru_cache
from covid_19_keywords import KEYWORDS

logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')

class ProcessTweet():
    """Wrapper class for functions to process/modify tweets"""

    def __init__(self, tweet=None, map_data=None, gc=None):
        self.tweet = tweet
        self.extended_tweet = self._get_extended_tweet()
        self.html_parser = HTMLParser()
        self.control_char_regex = r'[\r\n\t]+'
        self.map_data = map_data
        self.gc = gc

    @property
    def id(self):
        return self.tweet['id_str']

    @property
    def retweeted_status_id(self):
        return self.tweet['retweeted_status']['id_str']

    @property
    def retweeted_user_id(self):
        return self.tweet['retweeted_status']['user']['id_str']

    @property
    def quoted_status_id(self):
        return self.tweet['quoted_status']['id_str']

    @property
    def quoted_user_id(self):
        return self.tweet['quoted_status']['user']['id_str']

    @property
    def replied_status_id(self):
        return self.tweet['in_reply_to_status_id_str']

    @property
    def replied_user_id(self):
        return self.tweet['in_reply_to_user_id_str']

    @property
    def is_retweet(self):
        return 'retweeted_status' in self.tweet

    @property
    def has_quote(self):
        return 'quoted_status' in self.tweet

    @property
    def is_reply(self):
        return self.tweet['in_reply_to_status_id_str'] is not None

    @property
    def user_id(self):
        return self.tweet['user']['id_str']

    @property
    def user_timezone(self):
        try:
            return self.tweet['user']['timezone']
        except KeyError:
            return None

    @property
    def is_verified(self):
        return self.tweet['user']['verified'] is True

    @property
    def has_coordinates(self):
        return 'coordinates' in self.tweet and self.tweet['coordinates'] is not None

    @property
    def has_place(self):
        return self.tweet['place'] is not None

    @property
    def has_place_bounding_box(self):
        if not self.has_place:
            return False
        try:
            self.tweet['place']['bounding_box']['coordinates'][0]
        except (KeyError, TypeError):
            return False
        else:
            return True

    @lru_cache(maxsize=1)
    def get_text(self):
        """Get full text"""
        tweet_obj = self.tweet
        if self.is_retweet:
            # in retweets text is usually truncated, therefore get the text from original status
            tweet_obj = self.tweet['retweeted_status']
        if 'extended_tweet' in tweet_obj:
            text = tweet_obj['extended_tweet']['full_text']
        else:
            text = tweet_obj['text']
        return self.normalize_str(text)

    def convert_to_iso_time(self, date):
        ts = pd.to_datetime(date)
        return ts.isoformat()

    def extract(self, tweet_type='original'):
        geo_obj = self.get_geo_info()
        return {
                'id': self.id,
                'text': self.get_text(),
                'in_reply_to_status_id': self.replied_status_id,
                'in_reply_to_user_id': self.replied_user_id,
                'quoted_user_id': self.quoted_user_id if self.has_quote else None,
                'quoted_status_id': self.quoted_status_id if self.has_quote else None,
                'retweeted_user_id': self.retweeted_user_id if self.is_retweet else None,
                'retweeted_status_id': self.retweeted_status_id if self.is_retweet else None,
                'created_at': self.convert_to_iso_time(self.tweet['created_at']),
                'entities.user_mentions': self.get_user_mentions(),
                'user.id': self.user_id,
                'user.screen_name': self.tweet['user']['screen_name'],
                'user.name': self.tweet['user']['name'],
                'user.description': self.normalize_str(self.tweet['user']['description']),
                'user.timezone': self.user_timezone,
                'user.location': self.tweet['user']['location'],
                'user.num_followers': self.tweet['user']['followers_count'],
                'user.num_following': self.tweet['user']['friends_count'],
                'user.created_at': self.convert_to_iso_time(self.tweet['user']['created_at']),
                'user.statuses_count': self.tweet['user']['statuses_count'],
                'user.is_verified': self.is_verified,
                'lang': self.tweet['lang'],
                'token_count': self.get_token_count(),
                'is_retweet': self.is_retweet,
                'has_quote': self.has_quote,
                'is_reply': self.is_reply,
                'matched_keywords': self.matched_keywords(),
                **geo_obj
                }

    def matched_keywords(self):
        text = self.get_text()
        return [i for i, keyword in enumerate(KEYWORDS) if keyword.lower() in text]

    def normalize_str(self, s):
        if not s:
            return ''
        if not isinstance(s, str):
            s = str(s)
        # replace \t, \n and \r characters by a whitespace
        s = re.sub(self.control_char_regex, ' ', s)
        # replace HTML codes for new line characters
        s = s.replace('&#13;', '').replace('&#10;', '')
        s = self.html_parser.unescape(s)
        # remove duplicate whitespaces
        s = ' '.join(s.split())
        # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != 'C')

    def get_geo_info(self):
        """
        Tries to infer differen types of geoenrichment from tweet (ProcessTweet object)
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
            return self.map_data[self.map_data['ISO_A2'] == country_code].iloc[0].REGION_WB

        def get_subregion_by_country_code(country_code):
            return self.map_data[self.map_data['ISO_A2'] == country_code].iloc[0].SUBREGION

        def get_country_code_by_coords(longitude, latitude):
            coordinates = shapely.geometry.point.Point(longitude, latitude)
            within = self.map_data.geometry.apply(lambda p: coordinates.within(p))
            if sum(within) > 0:
                return self.map_data[within].iloc[0].ISO_A2
            else:
                dist = self.map_data.geometry.apply(lambda poly: poly.distance(coordinates))
                closest_country = self.map_data.iloc[dist.argmin()].ISO_A2
                logger.warning(f'Coordinates {longitude}, {latitude} were outside of a country land area but were matched to closest country ({closest_country})')
                return closest_country

        def convert_to_polygon(s):
            for i, _s in enumerate(s):
                s[i] = [float(_s[0]), float(_s[1])]
            return shapely.geometry.Polygon(s)

        geo_obj = {
                'longitude': None,
                'latitude': None,
                'country_code': None,
                'region': None,
                'subregion': None,
                'geo_type': 0
                }
        if self.has_coordinates:
            # try to get geo data from coordinates (<0.1% of tweets)
            geo_obj['longitude'] = self.tweet['coordinates']['coordinates'][0]
            geo_obj['latitude'] = self.tweet['coordinates']['coordinates'][1]
            geo_obj['country_code'] = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
            geo_obj['geo_type'] = 1
        elif self.has_place_bounding_box:
            # try to get geo data from place (roughly 1% of tweets)
            p = convert_to_polygon(self.tweet['place']['bounding_box']['coordinates'][0])
            geo_obj['longitude'] = p.centroid.x
            geo_obj['latitude'] = p.centroid.y
            country_code = self.tweet['place']['country_code']
            if country_code == '':
                # sometimes places don't contain country codes, try to resolve from coordinates
                country_code = get_country_code_by_coords(geo_obj['longitude'], geo_obj['latitude'])
            geo_obj['country_code'] = country_code
            geo_obj['geo_type'] = 2
        else:
            # try to parse user location
            locations = self.gc.decode(self.tweet['user']['location'])
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
            if geo_obj['country_code'] in self.map_data.ISO_A2.tolist():
                geo_obj['region'] = get_region_by_country_code(geo_obj['country_code'])
                geo_obj['subregion'] = get_subregion_by_country_code(geo_obj['country_code'])
            else:
                logger.warning(f'Unknown country_code {geo_obj["country_code"]}')
        return geo_obj

    def get_user_mentions(self):
        user_mentions = []
        if 'user_mentions' in self.extended_tweet['entities']:
            for mention in self.extended_tweet['entities']['user_mentions']:
                user_mentions.append(mention['id_str'])
        if len(user_mentions) == 0:
            return None
        else:
            return user_mentions

    def get_token_count(self):
        text = self.get_text()
        # remove user handles and URLs from text
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
        text = re.sub('(\@[^\s]+)', '', text)
        text = text.strip()
        doc = nlp(text, disable=['parser', 'tagger', 'ner'])
        # Count the number of tokens excluding stopwords
        token_count = len([token for token in doc if token.is_alpha and not token.is_stop])
        return token_count

    def extract_user(self):
        return {
                'user.id': self.user_id,
                'user.screen_name': self.tweet['user']['screen_name'],
                'user.name': self.tweet['user']['name'],
                'user.description': self.normalize_str(self.tweet['user']['description']),
                'user.timezone': self.user_timezone,
                'user.location': self.tweet['user']['location'],
                'user.num_followers': self.tweet['user']['followers_count'],
                'user.num_following': self.tweet['user']['friends_count'],
                'user.created_at': self.convert_to_iso_time(self.tweet['user']['created_at']),
                'user.statuses_count': self.tweet['user']['statuses_count'],
                'user.is_verified': self.is_verified
                }

    # private methods

    def _get_extended_tweet(self):
        if 'extended_tweet' in self.tweet:
            return self.tweet['extended_tweet']
        else:
            return self.tweet
