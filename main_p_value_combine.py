import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

def pearsonr_p_value_combine(column1, combined_column):
    valid_indices = ~np.isnan(column1) & ~np.isnan(combined_column)
    column1_cleaned = column1[valid_indices]
    combined_column_cleaned = combined_column[valid_indices]
    correlation_coefficient, p_value = pearsonr(column1_cleaned, combined_column_cleaned)
    print(f"Pearson correlation coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")

def normalize_columns(*columns):
    scaler = MinMaxScaler()
    normalized_columns = [scaler.fit_transform(column.reshape(-1, 1)).flatten() for column in columns]
    return normalized_columns

if __name__ == '__main__':
    column_auc = np.array([64.45, 67.82, 68.8, 71.79, 66.4, 70.79, 71.77, 74.78, 75.35, 71.85])
    column_time = np.array([2.6, 2.39, 2.13, 2.44, 2.5, 2.27, 2.39, 2.35, 2.32, 2.42])
    column_modify = np.array([0.17, 0.18, 0.15, 0.17, 0.16, 0.14, 0.17, 0.18, 0.14, 0.16])
    column_review = np.array([1.82, 1.69, 1.62, 1.67, 1.71, 1.66, 1.68, 1.69, 1.61, 1.7])
    print('The correlation and p-value between AUC and the composite metric (diagnosis time, number of diagnosis modifications, and number of EHR views)')

    norm_time, norm_modify, norm_review = normalize_columns(column_time, column_modify, column_review)
    combined_multiply_norm = norm_time * norm_modify * norm_review
    print("The result of multiplying after normalization：")
    pearsonr_p_value_combine(column_auc, combined_multiply_norm)
