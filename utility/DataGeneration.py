import os
import pandas as pd


def dataset_foreach_feature(csv_file):
    current_path = os.getcwd()
    csv_path = os.path.join(current_path, csv_file)
    dataset = pd.read_csv(csv_path, sep='\t')
    print(f'Testing {csv_file} for binary classification: BL0 and PA4.')

    # Data standardization (z-score [-1,1], alternatively we could use minmax)
    output_labels = dataset['PAIN_LEVEL'].copy()
    raw_input = dataset.drop(['SUBJECT_ID', 'PAIN_LEVEL', 'AGE', 'TRIAL', 'SEX'], axis=1)
    raw_input = (raw_input - raw_input.mean()) / raw_input.std()
    raw_input = raw_input.dropna(axis=1)
    raw_input = raw_input.dropna(axis=0)

    gsr = raw_input.iloc[:, :7]
    ecg = raw_input.iloc[:, 7:14]
    emg_trap = raw_input.iloc[:, 14:21]
    emg_corr = raw_input.iloc[:, 21:28]
    emg_zyg = raw_input.iloc[:, 28:35]

    return gsr, ecg, emg_trap, emg_corr, emg_zyg, output_labels


def io_sets_multiclass(csv_file):
    # Data loading
    current_path = os.getcwd()
    csv_path = os.path.join(current_path, csv_file)
    dataset = pd.read_csv(csv_path, sep='\t')
    print(f'Extracting sets from {csv_file}, for multiclass classification of the labels BL0 and PA4.')

    # Mapping to numeric values (we could use the one-hot encoding)
    dataset['PAIN_LEVEL'] = dataset['PAIN_LEVEL'].replace(['BL1', 'PA1', 'PA2', 'PA3', 'PA4'], [0, 1, 2, 3, 4])

    # Drop the null attributes
    dataset = dataset.dropna(axis=1)

    # Remove the subject column (useless)
    dataset = dataset.drop(['SUBJECT_ID'], axis=1)

    # Dataset correlation (extra step)
    correlation = dataset.drop(['TRIAL', 'AGE'], axis=1).corr(numeric_only=True)
    correlation["PAIN_LEVEL"].sort_values(ascending=False)

    # Data standardization (z-score [-1,1], alternatively we could use minmax)
    y = dataset['PAIN_LEVEL'].copy()
    X = dataset.drop(['PAIN_LEVEL', 'AGE', 'TRIAL'], axis=1)
    X = (X - X.mean(numeric_only=True)) / X.std(numeric_only=True)
    X = X.dropna(axis=1)
    X = X.dropna(axis=0)

    return X, y, correlation


def io_sets_binary(csv_file):
    current_path = os.getcwd()
    csv_path = os.path.join(current_path, csv_file)
    dataset = pd.read_csv(csv_path, sep='\t')

    # Mapping to numeric values (we could use the one-hot encoding)
    dataset['PAIN_LEVEL'] = dataset['PAIN_LEVEL'].replace(['BL1', 'PA4'], [0, 1])

    # Drop the null attributes
    dataset = dataset.dropna(axis=1)

    # Remove the subject column (useless)
    dataset = dataset.drop(['SUBJECT_ID'], axis=1)

    # Data standardization (z-score [-1,1], alternatively we could use minmax)
    y = dataset['PAIN_LEVEL'].copy()
    X = dataset.drop(['PAIN_LEVEL', 'AGE', 'TRIAL', 'SEX'], axis=1)
    X = (X - X.mean(numeric_only=True)) / X.std(numeric_only=True)
    X = X.dropna(axis=1)
    X = X.dropna(axis=0)

    return X, y


def io_sets_ecg_only():
    csv_file = "features/extracted_bio_features_binary.csv"
    current_path = os.getcwd()
    csv_path = os.path.join(current_path, csv_file)
    dataset = pd.read_csv(csv_path, sep='\t')

    # Mapping to numeric values (we could use the one-hot encoding)
    dataset['PAIN_LEVEL'] = dataset['PAIN_LEVEL'].replace(['BL1', 'PA4'], [0, 1])

    # Drop the null attributes
    dataset = dataset.dropna(axis=1)

    # Remove the subject column (useless)
    dataset = dataset.drop(['SUBJECT_ID'], axis=1)

    # Data standardization (z-score [-1,1], alternatively we could use minmax)
    y = dataset['PAIN_LEVEL'].copy()
    X = dataset.drop(['PAIN_LEVEL', 'AGE', 'TRIAL', 'SEX'], axis=1)

    # Drop all the other features except
    X = (X - X.mean(numeric_only=True)) / X.std(numeric_only=True)
    X = X.dropna(axis=1)
    X = X.dropna(axis=0)
    X = X[['MEAN_ECG', 'MEDIAN_ECG', 'MAX_ECG', 'MIN_ECG', 'STD_ECG', 'SUM_ECG', 'VAR_ECG']]

    return X, y


def io_sets_gsr_only():
    csv_file = "features/extracted_bio_features_binary.csv"
    current_path = os.getcwd()
    csv_path = os.path.join(current_path, csv_file)
    dataset = pd.read_csv(csv_path, sep='\t')

    # Mapping to numeric values (we could use the one-hot encoding)
    dataset['PAIN_LEVEL'] = dataset['PAIN_LEVEL'].replace(['BL1', 'PA4'], [0, 1])

    # Drop the null attributes
    dataset = dataset.dropna(axis=1)

    # Remove the subject column (useless)
    dataset = dataset.drop(['SUBJECT_ID'], axis=1)

    # Data standardization (z-score [-1,1], alternatively we could use minmax)
    y = dataset['PAIN_LEVEL'].copy()
    X = dataset.drop(['PAIN_LEVEL', 'AGE', 'TRIAL', 'SEX'], axis=1)

    # Drop all the other features except
    X = (X - X.mean(numeric_only=True)) / X.std(numeric_only=True)
    X = X.dropna(axis=1)
    X = X.dropna(axis=0)
    X = X[['MEAN_GSR', 'MEDIAN_GSR', 'MAX_GSR', 'MIN_GSR', 'STD_GSR', 'SUM_GSR', 'VAR_GSR']]

    return X, y


def io_sets_video_only():
    csv_file = "features/extracted_video_features_binary.csv"
    current_path = os.getcwd()
    csv_path = os.path.join(current_path, csv_file)
    dataset = pd.read_csv(csv_path, sep='\t')

    # Mapping to numeric values (we could use the one-hot encoding)
    dataset['PAIN_LEVEL'] = dataset['PAIN_LEVEL'].replace(['BL1', 'PA4'], [0, 1])

    # Drop the null attributes
    dataset = dataset.dropna(axis=1)

    # Remove the subject column (useless)
    dataset = dataset.drop(['SUBJECT_ID'], axis=1)

    # Data standardization (z-score [-1,1], alternatively we could use minmax)
    y = dataset['PAIN_LEVEL'].copy()
    X = dataset.drop(['PAIN_LEVEL', 'AGE', 'TRIAL', 'SEX'], axis=1)

    # Drop all the other features except
    X = (X - X.mean(numeric_only=True)) / X.std(numeric_only=True)
    X = X.dropna(axis=1)
    X = X.dropna(axis=0)
    # X = X.drop(['HEAD_ROTATION_X', 'HEAD_ROTATION_Y', 'HEAD_ROTATION_Z'], axis=1)
    return X, y


def binarize_early_fusion_features():
    f = open('features/fused_features.csv', 'r', encoding='UTF-8', newline='')

    df = pd.read_csv(f, sep='\t')
    df = df[~df['PAIN_LEVEL'].isin(['PA1', 'PA2', 'PA3'])]

    df.to_csv('features/fused_features_binary.csv', index=False, sep='\t')
    f.close()


def early_fusion():
    f0 = open('features/extracted_bio_features.csv', 'r', encoding='UTF-8', newline='')
    f1 = open('features/extracted_video_features.csv', 'r', encoding='UTF-8', newline='')

    df1 = pd.read_csv(f1, sep='\t', decimal='.')
    df0 = pd.read_csv(f0, sep='\t', decimal='.')

    df_merged = pd.merge(df0, df1, how='inner', on=['SUBJECT_ID', 'TRIAL'], suffixes=('', '_drop'))
    df_merged.drop([col for col in df_merged.columns if 'drop' in col], axis=1, inplace=True)
    df_merged.to_csv('features/fused_features.csv', index=False, sep='\t')

    f1.close()
    f0.close()


def binarize_biosignals_and_video_features():
    f = open('features/extracted_bio_features.csv', 'r', encoding='UTF-8', newline='')
    df = pd.read_csv(f, sep='\t')
    df = df[~df['PAIN_LEVEL'].isin(['PA1', 'PA2', 'PA3'])]
    df.to_csv('features/extracted_bio_features_binary.csv', index=False, sep='\t')
    f.close()

    f = open('features/extracted_video_features.csv', 'r', encoding='UTF-8', newline='')
    df = pd.read_csv(f, sep='\t')
    df = df[~df['PAIN_LEVEL'].isin(['PA1', 'PA2', 'PA3'])]
    df.to_csv('features/extracted_video_features_binary.csv', index=False, sep='\t')
    f.close()
