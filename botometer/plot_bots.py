import pandas as pd
import os
import seaborn as sns
import sys
sys.path.append('..')
from utils.plot_helpers import save_fig
import matplotlib.pyplot as plt


data_dir = os.path.join('..', 'data')
annotation_data = os.path.join(data_dir, 'annotation_data', 'annotated_users_ten_languages-2.pickle')
botometer_data = os.path.join(data_dir, 'botometer', 'botometer_results_5000.jsonl')

def get_annotation_data():
    df = pd.read_pickle(annotation_data)
    df = df.rename(columns={'username': 'user.screen_name'})
    return df

def get_botometer_data():
    df = pd.read_json(botometer_data, lines=True)
    df = df.dropna(subset=['user'])
    df['user.id'] = df.user.apply(lambda s: s['id_str'])
    df['user.screen_name'] = df.user.apply(lambda s: s['screen_name'])
    df['cap_universal'] = df.cap.apply(lambda s: s['universal'])
    df['cap_english'] = df.cap.apply(lambda s: s['english'])
    df['score_english'] = df.scores.apply(lambda s: s['english'])
    df['score_universal'] = df.scores.apply(lambda s: s['universal'])
    df = df.drop(columns=['user', 'cap', 'scores', 'display_scores', 'categories', 'error'])
    return df

def main():
    df_annot = get_annotation_data()
    df_bot = get_botometer_data()

    df = df_bot.merge(df_annot, on='user.screen_name')
    df = df[['user.screen_name', 'cap_universal', 'n_covid_tweeting_language', 'majority_vote_category']]
    df = df.melt(id_vars=['user.screen_name', 'n_covid_tweeting_language', 'majority_vote_category'], var_name='score', value_name='cap_score')

    v = 2
    sns.catplot(x="cap_score", y="n_covid_tweeting_language", kind='box', data=df, aspect=2)
    save_fig(plt.gcf(), 'bot_distribution', v)

if __name__ == "__main__":
    main()
