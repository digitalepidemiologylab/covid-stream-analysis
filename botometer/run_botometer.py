import botometer
import os
import pandas as pd
import json
from tqdm import tqdm
import logging
import secrets

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

twitter_app_auth = {
    'consumer_key': secrets.consumer_key,
    'consumer_secret': secrets.consumer_secret,
    'access_token': secrets.access_token,
    'access_token_secret': secrets.access_token_secret
  }
bom = botometer.Botometer(wait_on_ratelimit=True, rapidapi_key=secrets.rapidapi_key, **twitter_app_auth)
data_dir = os.path.join('..', 'data')
annotation_data = os.path.join(data_dir, 'annotation_data', 'annotated_users_ten_languages-2.pickle')

def get_usernames():
    df = pd.read_pickle(annotation_data)
    df = df.dropna(subset=['username'])
    df = df.drop_duplicates(subset=['username'])
    usernames = df.username.sample(5000)
    return usernames

def main():
    usernames = get_usernames()
    f_out = os.path.join(data_dir, 'botometer', 'botometer_results_5000.jsonl')
    for screen_name, result in tqdm(bom.check_accounts_in(usernames), total=len(usernames)):
        with open(f_out, 'a') as f:
            f.write(json.dumps(result) + '\n') 

if __name__ == "__main__":
    main()
