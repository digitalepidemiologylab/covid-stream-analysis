import pandas as pd
import os
import seaborn as sns
import sys
sys.path.append('..')
from utils.plot_helpers import save_fig
import matplotlib.pyplot as plt
import scipy.stats


data_dir = os.path.join('..', 'data')
annotation_data = os.path.join(data_dir, 'annotation_data', 'annotated_users_ten_languages-2.pickle')
botometer_data = os.path.join(data_dir, 'botometer')

def get_annotation_data():
    df = pd.read_pickle(annotation_data)
    df = df.rename(columns={'username': 'user.screen_name'})
    return df

def get_botometer_data(data_type='annotation', num=5000):
    f_path = os.path.join(botometer_data, f'botometer_results_{num}_{data_type}.jsonl')
    df = pd.read_json(f_path, lines=True)
    df = df.dropna(subset=['user'])
    df['user.id'] = df.user.apply(lambda s: s['id_str'])
    df['user.screen_name'] = df.user.apply(lambda s: s['screen_name'])
    df['cap_universal'] = df.cap.apply(lambda s: s['universal'])
    df['cap_english'] = df.cap.apply(lambda s: s['english'])
    df['score_english'] = df.scores.apply(lambda s: s['english'])
    df['score_universal'] = df.scores.apply(lambda s: s['universal'])
    df['is_bot'] = df['cap_universal'] > .25
    df = df.drop(columns=['user', 'cap', 'scores', 'display_scores', 'categories', 'error'])
    return df

def main():
    df_annot = get_annotation_data()
    df_bot_annot = get_botometer_data(data_type='annotation')
    df_bot_raw = get_botometer_data(data_type='raw')
    df = df_bot_annot.merge(df_annot, on='user.screen_name')

    for col in ['majority_vote_category', 'majority_vote_type']:
        num_bots = df.groupby(col).is_bot.sum()
        num_total = df.groupby(col).is_bot.count()
        num_tests = len(num_total)
        mean_bot = df.is_bot.mean()

        for is_bot, total, label in zip(num_bots, num_total, df.groupby(col).groups.keys()):
            test = scipy.stats.binom_test(is_bot, total, mean_bot, alternative='greater')
            is_sig = test < (.05/num_tests)
            print(f'Label {label}: {test:.3f} (is_sig: {is_sig}), fraction: {is_bot/total:.3f}')

if __name__ == "__main__":
    main()
