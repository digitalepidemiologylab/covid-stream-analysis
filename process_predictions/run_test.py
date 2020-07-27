import pandas as pd
import json
import os
import sklearn.metrics
import shutil

test_folder = os.path.join('..', 'data', 'annotation_data', 'test')
predict_folder = os.path.join('..', '..', 'covid-twitter-bert', 'data', 'predictions')
output_folder = os.path.join('..', 'text_classification', 'output')

# drwxrwxr-x  3 martin 4.0K Jul 23 21:45 predictions_2020-07-23_19-27-28_359106_category_multilang_no_pt/
# drwxrwxr-x  3 martin 4.0K Jul 23 21:45 predictions_2020-07-23_19-31-00_051241_type_multilang_no_pt/
# drwxrwxr-x  3 martin 4.0K Jul 23 21:45 predictions_2020-07-23_19-38-10_753655_category_no_pt/
# drwxrwxr-x  3 martin 4.0K Jul 23 21:45 predictions_2020-07-23_19-39-52_580919_type_no_pt/
#

paths = {
        'category_multilang': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-21_20-37-56_033749', 'predictions', 'category_merged_multilang.jsonl'),
            'test_data': os.path.join(test_folder, 'category_merged_multilang', 'all.csv')
            },
        'type_multilang': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-21_20-41-43_800266', 'predictions', 'type_merged_multilang.jsonl'),
            'test_data': os.path.join(test_folder, 'type_merged_multilang', 'all.csv') 
            },
        'category_multilang_no_pt': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-23_19-27-28_359106_category_multilang_no_pt', 'predictions', 'category_merged_multilang.jsonl'),
            'test_data': os.path.join(test_folder, 'category_merged_multilang', 'all.csv')
            },
        'type_multilang_no_pt': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-23_19-31-00_051241_type_multilang_no_pt', 'predictions', 'type_merged_multilang.jsonl'),
            'test_data': os.path.join(test_folder, 'type_merged_multilang', 'all.csv') 
            },
        'category_no_pt': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-23_19-38-10_753655_category_no_pt', 'predictions', 'category_merged.jsonl'),
            'test_data': os.path.join(test_folder, 'category_merged', 'all.csv')
            },
        'type_no_pt': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-23_19-39-52_580919_type_no_pt', 'predictions', 'type_merged.jsonl'),
            'test_data': os.path.join(test_folder, 'type_merged', 'all.csv') 
            },
        'category': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-22_23-22-04_469275_category', 'predictions', 'category_merged.jsonl'),
            'test_data': os.path.join(test_folder, 'category_merged', 'all.csv')
            },
        'type': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-22_23-17-29_770500_type', 'predictions', 'type_merged.jsonl'),
            'test_data': os.path.join(test_folder, 'type_merged', 'all.csv') 
            },
        'category_unambiguous_multilang': {
            'predict_data': os.path.join(predict_folder, 'predictions_2020-07-24_14-09-30_484704_category_unambiguous_multilang', 'predictions', 'category_merged_unambiguous_multilang.jsonl'),
            'test_data': os.path.join(test_folder, 'category_merged_unambiguous_multilang', 'all.csv') 
            }
        }

def performance_metrics(y_true, y_pred, metrics=None, averaging=None, label_mapping=None):
    """
    Compute performance metrics
    """
    def _compute_performance_metric(scoring_function, m, y_true, y_pred):
        # compute averaging
        for av in averaging:
            if av is None:
                metrics_by_class = scoring_function(y_true, y_pred, average=av, labels=labels)
                for i, class_metric in enumerate(metrics_by_class):
                    if label_mapping is None:
                        label_name = labels[i]
                    else:
                        label_name = label_mapping[labels[i]]
                    scores['scores_by_label'][m + '_' + str(label_name)] = class_metric
            else:
                scores[m + '_' + av] = scoring_function(y_true, y_pred, average=av, labels=labels)

    if averaging is None:
        averaging = ['micro', 'macro', 'weighted', None]
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef', 'cohen_kappa']
    scores = {'scores_by_label': {}}
    labels = sorted(list(set(y_true).union(set(y_pred))))
    if len(labels) <= 2:
        # binary classification
        averaging += ['binary']
    for m in metrics:
        if m == 'accuracy':
            scores[m] = sklearn.metrics.accuracy_score(y_true, y_pred)
        elif m == 'precision':
            _compute_performance_metric(sklearn.metrics.precision_score, m, y_true, y_pred)
        elif m == 'recall':
            _compute_performance_metric(sklearn.metrics.recall_score, m, y_true, y_pred)
        elif m == 'f1':
            _compute_performance_metric(sklearn.metrics.f1_score, m, y_true, y_pred)
        elif m == 'matthews_corrcoef':
            scores[m] = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
        elif m == 'cohen_kappa':
            scores[m] = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    return scores

def main():
    for dataset, _p in paths.items():
        df_test = pd.read_csv(_p['test_data'])
        df_predict = pd.read_json(_p['predict_data'], lines=True)
        df_predict = df_predict.rename(columns={'label': 'prediction'})
        df = pd.concat([df_test, df_predict], axis=1)
        scores = performance_metrics(df.label, df.prediction)
        run_name = f'test_{dataset}'
        f_output = os.path.join(output_folder, run_name)
        if os.path.isdir(f_output):
            shutil.rmtree(f_output)
        os.makedirs(f_output)
        print(f'Writing to {f_output}')
        with open(os.path.join(f_output, 'test_output.json'), 'w') as f:
            json.dump(scores, f, indent=4)
        df[['text', 'label', 'prediction']].to_csv(os.path.join(f_output, 'test_output.csv'))
        with open(os.path.join(f_output, 'run_config.json'), 'w') as f:
            json.dump({'name': run_name}, f, indent=4)

if __name__ == "__main__":
    main()
