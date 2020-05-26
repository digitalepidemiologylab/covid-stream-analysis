import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ExtractTweet():
    """Wrapper class for functions to process/modify tweets"""

    def __init__(self, tweet):
        self.tweet = tweet

    @property
    def id(self):
        return self.tweet['id_str']

    @property
    def retweeted_status_id(self):
        return self.tweet['retweeted_status']['id_str']

    @property
    def quoted_status_id(self):
        return self.tweet['quoted_status']['id_str']

    @property
    def replied_status_id(self):
        return str(self.tweet['in_reply_to_status_id'])

    @property
    def is_retweet(self):
        return 'retweeted_status' in self.tweet

    @property
    def has_quoted_status(self):
        return 'quoted_status' in self.tweet

    @property
    def is_reply(self):
        return self.tweet['in_reply_to_status_id'] is not None

    @property
    def user_id(self):
        return self.tweet['user']['id_str']

    @property
    def text(self):
        """Get full text"""
        if 'extended_tweet' in self.tweet:
            return self.tweet['extended_tweet']['full_text']
        else:
            return self.tweet['text']

    @property
    def user_timezone(self):
        try:
            return self.tweet['user']['timezone']
        except KeyError:
            return None

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

    def convert_to_iso_time(self, date):
        ts = pd.to_datetime(date)
        return ts.isoformat()


    def extract_fields(self):
        return {
                'id': self.id,
                'text': self.text,
                'created_at': self.convert_to_iso_time(self.tweet['created_at']),
                'user.id': self.user_id,
                'user.screen_name': self.tweet['user']['screen_name'],
                'user.name': self.tweet['user']['name'],
                'user.description': self.tweet['user']['description'],
                'user.timezone': self.user_timezone,
                'user.location': self.tweet['user']['location'],
                'user.num_followers': self.tweet['user']['followers_count'],
                'user.num_following': self.tweet['user']['friends_count'],
                'user.created_at': self.convert_to_iso_time(self.tweet['user']['created_at']),
                'user.statuses_count': self.tweet['user']['statuses_count'],
                'lang': self.tweet['lang']
                }
