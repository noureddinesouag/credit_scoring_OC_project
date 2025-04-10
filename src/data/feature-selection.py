import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
import gc
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    """Context manager for timing code execution."""
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")

def load_data(train_path, test_path=None):
    """Load the processed data."""
    print("Loading processed data...")
    train_df = pd.read_pickle(train_path)
    if test_path:
        test_df = pd.read_pickle(test_path)
        return train_df, test_df
    return train_df

def remove_high_missing(df, threshold=0.9):
    """Remove features with high missing values percentage."""
    print(f"Removing features with >{threshold*100:.0f}% missing values...")
    missing_series = df.isnull().sum() / len(df)
    missing_stats = pd.DataFrame({'column': missing_series.index, 'percent_missing': missing_series.values})
    high_missing = missing_stats[missing_stats.percent_missing > threshold]['column'].tolist()
    print(f"Removed {len(high_missing)} features with high missing values")
    return df.drop(columns=high_missing), high_missing

def remove_high_correlation(df, target_col, threshold=0.95):
    """Remove highly correlated features."""
    print(f"Removing highly correlated features (threshold={threshold})...")
    # Exclude the target column
    df_corr = df.drop(columns=[target_col]).corr().abs()
    
    # Create a mask for the upper triangle
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Removed {len(to_drop)} highly correlated features")
    return df.drop(columns=to_drop), to_drop

def remove_low_variance(df, target_col, threshold=0.01):
    """Remove features with low variance."""
    print(f"Removing low variance features (threshold={threshold})...")
    # Exclude the target column
    df_var = df.drop(columns=[target_col])
    
    # Initialize the selector
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df_var)
    
    # Get feature mask and feature names
    feature_mask = selector.get_support()
    low_variance = [column for column, selected in zip(df_var.columns, feature_mask) if not selected]
    
    print(f"Removed {len(low_variance)} low variance features")
    return df.drop(columns=low_variance), low_variance

def select_by_univariate(df, target_col, k=100, method='f_classif'):
    """Select top k features based on univariate statistical tests."""
    print(f"Selecting top {k} features using {method}...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Choose the scoring method
    if method == 'f_classif':
        score_func = f_classif
    elif method == 'mutual_info':
        score_func = mutual_info_classif
    else:
        raise ValueError("method must be one of 'f_classif' or 'mutual_info'")
    
    # Initialize and fit the selector
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    
    # Get feature mask and feature names
    feature_mask = selector.get_support()
    selected_features = X.columns[feature_mask].tolist()
    
    # Create a dataframe with feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    print(f"Selected {len(selected_features)} features using {method}")
    return df[selected_features + [target_col]], selected_features, feature_scores

def select_by_model_importance(df, target_col, k=100, model_type='random_forest'):
    """Select top k features based on model feature importance."""
    print(f"Selecting top {k} features using {model_type} importance...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Choose the model type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError("model_type must be one of 'random_forest' or 'lightgbm'")
    
    # Train the model
    model.fit(X, y)
    
    # Get feature importances
    if model_type == 'random_forest':
        importances = model.feature_importances_
    elif model_type == 'lightgbm':
        importances = model.feature_importances_
    
    # Create a dataframe with feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    
    # Select top k features
    top_features = feature_importances.head(k)['Feature'].tolist()
    
    print(f"Selected {len(top_features)} features using {model_type} importance")
    return df[top_features + [target_col]], top_features, feature_importances

def forward_selection(df, target_col, k=100, model_type='lightgbm', cv=5, metric='roc_auc', max_candidates=20):
    """
    Perform forward feature selection.
    
    Start with an empty feature set and add the feature that improves performance the most,
    until k features are selected or no improvement is observed.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with features and target.
    target_col : str
        Name of the target column.
    k : int
        Maximum number of features to select.
    model_type : str
        Type of model to use for selection ('lightgbm' or 'random_forest').
    cv : int
        Number of cross-validation folds.
    metric : str
        Evaluation metric for cross-validation.
    max_candidates : int
        Number of candidate features to consider at each step.
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with selected features and target.
    list
        List of selected feature names.
    """
    print(f"Performing forward selection up to {k} features using {model_type}...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Choose the model type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                   objective='binary', class_weight='balanced')
    else:
        raise ValueError("model_type must be one of 'random_forest' or 'lightgbm'")
    
    # Initialize variables
    selected_features = []
    unselected_features = list(X.columns)
    cv_scores = []
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for i in range(min(k, len(X.columns))):
        print(f"Selection step {i+1}/{min(k, len(X.columns))}")
        
        # If we've already selected features, evaluate current performance
        if selected_features:
            X_selected = X[selected_features]
            current_score = np.mean(cross_val_score(model, X_selected, y, cv=skf, scoring=metric, n_jobs=-1))
            print(f"Current score with {len(selected_features)} features: {current_score:.5f}")
        else:
            current_score = 0
        
        best_score = current_score
        best_feature = None
        
        # Get candidate features (either all remaining or the top ones by univariate score)
        if len(unselected_features) > max_candidates:
            # Use univariate importance to limit candidates
            selector = SelectKBest(score_func=f_classif, k=max_candidates)
            X_unselected = X[unselected_features]
            selector.fit(X_unselected, y)
            
            # Get feature mask and feature names
            feature_mask = selector.get_support()
            candidates = [feature for feature, selected in zip(unselected_features, feature_mask) if selected]
        else:
            candidates = unselected_features
        
        print(f"Testing {len(candidates)} candidate features")
        
        # Evaluate each candidate feature
        for feature in candidates:
            # Add this feature to the selected features
            current_features = selected_features + [feature]
            X_current = X[current_features]
            
            # Evaluate performance
            try:
                scores = cross_val_score(model, X_current, y, cv=skf, scoring=metric, n_jobs=-1)
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            except Exception as e:
                print(f"Error evaluating feature {feature}: {e}")
        
        # If we found a feature that improves performance, add it
        if best_feature and best_score > current_score:
            selected_features.append(best_feature)
            unselected_features.remove(best_feature)
            cv_scores.append(best_score)
            print(f"Added feature: {best_feature}, New score: {best_score:.5f}")
        else:
            print("No improvement found, stopping selection")
            break
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o')
    plt.title('Forward Selection Learning Curve')
    plt.xlabel('Number of Features')
    plt.ylabel(f'Cross-Validation {metric.upper()}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forward_selection_curve.png')
    plt.close()
    
    print(f"Selected {len(selected_features)} features using forward selection")
    return df[selected_features + [target_col]], selected_features

def recursive_feature_elimination(df, target_col, model_type='lightgbm', 
                                step_size=0.2, min_features=50, cv=5, metric='roc_auc'):
    """
    Perform recursive feature elimination.
    
    Start with all features and recursively remove the least important ones,
    evaluating model performance at each step.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with features and target.
    target_col : str
        Name of the target column.
    model_type : str
        Type of model to use for selection ('lightgbm' or 'random_forest').
    step_size : float
        Fraction of features to remove at each step.
    min_features : int
        Minimum number of features to keep.
    cv : int
        Number of cross-validation folds.
    metric : str
        Evaluation metric for cross-validation.
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with selected features and target.
    list
        List of selected feature names.
    """
    print(f"Performing recursive feature elimination using {model_type}...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Choose the model type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                   objective='binary', class_weight='balanced')
    else:
        raise ValueError("model_type must be one of 'random_forest' or 'lightgbm'")
    
    # Initialize variables
    current_features = list(X.columns)
    cv_scores = []
    n_features = []
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    while len(current_features) > min_features:
        print(f"Current feature count: {len(current_features)}")
        
        # Evaluate current feature set
        X_current = X[current_features]
        scores = cross_val_score(model, X_current, y, cv=skf, scoring=metric, n_jobs=-1)
        current_score = np.mean(scores)
        cv_scores.append(current_score)
        n_features.append(len(current_features))
        print(f"Score with {len(current_features)} features: {current_score:.5f}")
        
        # Train model on full data to get feature importances
        model.fit(X_current, y)
        
        # Get feature importances
        if model_type == 'random_forest':
            importances = model.feature_importances_
        elif model_type == 'lightgbm':
            importances = model.feature_importances_
        
        # Create a dataframe with feature importances
        feature_importances = pd.DataFrame({
            'Feature': current_features,
            'Importance': importances
        })
        feature_importances = feature_importances.sort_values('Importance')
        
        # Calculate number of features to remove
        n_to_remove = max(1, int(len(current_features) * step_size))
        n_to_remove = min(n_to_remove, len(current_features) - min_features)
        
        if n_to_remove <= 0:
            break
        
        # Remove the least important features
        features_to_remove = feature_importances.head(n_to_remove)['Feature'].tolist()
        current_features = [f for f in current_features if f not in features_to_remove]
        print(f"Removed {n_to_remove} features")
    
    # Find the best performing feature set
    best_idx = np.argmax(cv_scores)
    best_score = cv_scores[best_idx]
    best_n_features = n_features[best_idx]
    
    print(f"Best score: {best_score:.5f} with {best_n_features} features")
    
    # Train model on full data to get feature importances for final selection
    model.fit(X, y)
    
    # Get feature importances
    if model_type == 'random_forest':
        importances = model.feature_importances_
    elif model_type == 'lightgbm':
        importances = model.feature_importances_
    
    # Create a dataframe with feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    
    # Select top features
    selected_features = feature_importances.head(best_n_features)['Feature'].tolist()
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_features, cv_scores, marker='o')
    plt.axvline(x=best_n_features, color='r', linestyle='--', 
                label=f'Best: {best_n_features} features, {best_score:.5f} score')
    plt.title('Recursive Feature Elimination Learning Curve')
    plt.xlabel('Number of Features')
    plt.ylabel(f'Cross-Validation {metric.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rfe_curve.png')
    plt.close()
    
    print(f"Selected {len(selected_features)} features using RFE")
    return df[selected_features + [target_col]], selected_features

def clean_feature_names(df, target_col=None):
    """
    Clean feature names to remove special characters that might cause issues with models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with feature names to clean.
    target_col : str, optional
        Name of the target column to preserve.
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with cleaned feature names.
    dict
        Mapping from original to cleaned feature names.
    """
    import re
    
    print("Cleaning feature names to remove special characters...")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # Function to clean a feature name
    def clean_name(name):
        # Replace special characters with underscores
        clean = re.sub(r'[^\w\s]', '_', name)
        # Replace spaces with underscores
        clean = re.sub(r'\s+', '_', clean)
        # Ensure name starts with a letter or underscore (not a number)
        if clean[0].isdigit():
            clean = 'f_' + clean
        return clean
    
    # Create mapping from original to cleaned names
    name_mapping = {}
    new_columns = []
    
    for col in df.columns:
        # Skip target column if specified
        if target_col and col == target_col:
            new_columns.append(col)
            continue
        
        # Clean the column name
        new_name = clean_name(col)
        
        # Handle duplicates by adding a suffix
        suffix = 1
        while new_name in new_columns:
            new_name = f"{clean_name(col)}_{suffix}"
            suffix += 1
        
        name_mapping[col] = new_name
        new_columns.append(new_name)
    
    # Rename columns
    df_clean.columns = new_columns
    
    print(f"Cleaned {len(name_mapping)} feature names")
    return df_clean, name_mapping

def revert_feature_names(df, name_mapping, exclude_cols=None):
    """
    Revert column names to their original names using the mapping.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with cleaned feature names.
    name_mapping : dict
        Mapping from original to cleaned feature names.
    exclude_cols : list, optional
        Columns to exclude from renaming.
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with original feature names.
    """
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in name_mapping.items()}
    
    # Create a copy of the dataframe
    df_original = df.copy()
    
    # For each column, revert to original name if in the mapping
    rename_dict = {}
    for col in df.columns:
        if exclude_cols and col in exclude_cols:
            continue
        if col in reverse_mapping:
            rename_dict[col] = reverse_mapping[col]
    
    # Rename columns
    df_original = df_original.rename(columns=rename_dict)
    
    return df_original

def select_optimal_features(train_df, test_df=None, target_col='TARGET'):
    """
    Main function to perform feature selection.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset with features and target.
    test_df : pandas.DataFrame, optional
        Testing dataset with features.
    target_col : str
        Name of the target column.
        
    Returns:
    --------
    pandas.DataFrame
        Training dataset with selected features.
    pandas.DataFrame or None
        Testing dataset with selected features, if provided.
    list
        List of selected feature names.
    """
    with timer("Feature selection"):
        # Clean feature names to avoid issues with LightGBM
        train_df_clean, name_mapping = clean_feature_names(train_df, target_col)
        if test_df is not None:
            test_df_clean, _ = clean_feature_names(test_df)
        
        # 1. Remove features with high missing values
        train_df_clean, high_missing = remove_high_missing(train_df_clean, threshold=0.9)
        
        # 2. Remove low variance features
        train_df_clean, low_variance = remove_low_variance(train_df_clean, target_col, threshold=0.01)
        
        # 3. Remove highly correlated features
        train_df_clean, high_corr = remove_high_correlation(train_df_clean, target_col, threshold=0.95)
        
        print(f"After basic cleaning: {train_df_clean.shape[1] - 1} features remaining")
        
        # 4. Univariate feature selection
        k_univariate = min(100, train_df_clean.shape[1] - 1)
        train_univariate, univariate_features, univariate_scores = select_by_univariate(
            train_df_clean, target_col, k=k_univariate, method='f_classif'
        )
        
        # 5. Model-based feature selection
        k_model = min(100, train_df_clean.shape[1] - 1)
        train_model, model_features, model_importances = select_by_model_importance(
            train_df_clean, target_col, k=k_model, model_type='lightgbm'
        )
        
        # 6. Forward selection - reduced to avoid long runtime
        k_forward = min(30, train_df_clean.shape[1] - 1)  # Reduced from 50 to 30
        train_forward, forward_features = forward_selection(
            train_df_clean, target_col, k=k_forward, model_type='lightgbm'
        )
        
        # 7. Recursive feature elimination - simplified
        k_rfe = min(50, train_df_clean.shape[1] - 1)
        # Use a simplified approach instead of full RFE
        _, rfe_features, _ = select_by_model_importance(
            train_df_clean, target_col, k=k_rfe, model_type='lightgbm'
        )
        
        # 8. Compare all methods
        methods = {
            'Univariate': univariate_features,
            'LightGBM Importance': model_features,
            'Forward Selection': forward_features,
            'Top Features (RFE)': rfe_features[:min(30, len(rfe_features))]  # Just use top features
        }
        
        comparison = compare_feature_selection_methods(train_df_clean, target_col, methods)
        
        # 9. Find the best method
        best_method = comparison.loc[comparison[f'Mean_roc_auc'].idxmax(), 'Method']
        best_model = comparison.loc[comparison[f'Mean_roc_auc'].idxmax(), 'Model']
        print(f"Best method: {best_method} with {best_model}")
        
        # 10. Get the selected features from the best method
        if best_method == 'Univariate':
            selected_features_clean = univariate_features
        elif best_method == 'LightGBM Importance':
            selected_features_clean = model_features
        elif best_method == 'Forward Selection':
            selected_features_clean = forward_features
        elif best_method == 'Top Features (RFE)':
            selected_features_clean = rfe_features[:min(30, len(rfe_features))]
        else:
            # Default to LightGBM importance
            selected_features_clean = model_features
        
        # Convert back to original feature names
        selected_features_original = []
        for feature in selected_features_clean:
            for original, cleaned in name_mapping.items():
                if cleaned == feature:
                    selected_features_original.append(original)
                    break
        
        # 11. Apply selection to train and test datasets
        train_selected = train_df[selected_features_original + [target_col]]
        
        if test_df is not None:
            test_selected = test_df[selected_features_original]
            return train_selected, test_selected, selected_features_original
        
        return train_selected, None, selected_features_original

def select_by_model_importance(df, target_col, k=100, model_type='random_forest'):
    """Select top k features based on model feature importance."""
    print(f"Selecting top {k} features using {model_type} importance...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Choose the model type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'lightgbm':
        # Configure LightGBM to handle potential issues
        model = lgb.LGBMClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            verbose=-1,  # Reduce verbosity
            importance_type='gain'  # Use gain for importance
        )
    else:
        raise ValueError("model_type must be one of 'random_forest' or 'lightgbm'")
    
    # Train the model
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a dataframe with feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    
    # Select top k features
    top_features = feature_importances.head(k)['Feature'].tolist()
    
    print(f"Selected {len(top_features)} features using {model_type} importance")
    return df[top_features + [target_col]], top_features, feature_importances

def compare_feature_selection_methods(df, target_col, methods=None, cv=5, metric='roc_auc'):
    """Compare different feature selection methods using cross-validation."""
    print(f"Comparing feature selection methods using {cv}-fold cross-validation...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if methods is None:
        return None
    
    # Initialize variables
    comparison = []
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Evaluate each method
    for method_name, selected_features in methods.items():
        print(f"Evaluating {method_name} with {len(selected_features)} features...")
        
        # Select features
        X_selected = X[selected_features]
        
        # Evaluate using different models
        for model_name, model in [
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('LightGBM', lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                          verbose=-1, importance_type='gain'))
        ]:
            try:
                # Evaluate performance
                scores = cross_val_score(model, X_selected, y, cv=skf, scoring=metric, n_jobs=-1)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Add to comparison
                comparison.append({
                    'Method': method_name,
                    'Model': model_name,
                    'Num_Features': len(selected_features),
                    f'Mean_{metric}': mean_score,
                    f'Std_{metric}': std_score
                })
                
                print(f"{method_name} with {model_name}: {mean_score:.5f} ± {std_score:.5f}")
            except Exception as e:
                print(f"Error evaluating {method_name} with {model_name}: {e}")
    
    # Convert to dataframe
    comparison_df = pd.DataFrame(comparison)
    
    # Plot the comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Method', y=f'Mean_{metric}', hue='Model', data=comparison_df)
    plt.title('Comparison of Feature Selection Methods')
    plt.xlabel('Method')
    plt.ylabel(f'Mean {metric.upper()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png')
    plt.close()
    
    return comparison_df

def save_important_features(features_df, output_file='important_features.csv'):
    """
    Save the feature importance dataframe to a CSV file.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        Dataframe with features and their importance.
    output_file : str
        Path to save the CSV file.
    """
    features_df.to_csv(output_file, index=False)
    print(f"Feature importances saved to {output_file}")

def plot_feature_importances(features_df, top_n=20, output_file='feature_importances.png'):
    """
    Plot the top N most important features.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        Dataframe with features and their importance.
    top_n : int
        Number of top features to plot.
    output_file : str
        Path to save the plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Get top N features
    top_features = features_df.head(top_n)
    
    # Create plot
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Feature importance plot saved to {output_file}")
def main():
    """Main function to run feature selection."""
    import os
    
    # Use correct relative paths based on the project structure
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the project root directory (assumed to be src/features)
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # File paths
    train_path = os.path.join(project_root, "data/processed/train_final.pkl")
    test_path = os.path.join(project_root, "data/processed/test_final.pkl")
    
    # Output directory
    output_dir = os.path.join(project_root, "data/features")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Verify paths exist
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            return
    
    # Load the data
    train_df, test_df = load_data(train_path, test_path)
    print(f"Loaded training data: {train_df.shape}")
    print(f"Loaded testing data: {test_df.shape}")
    
    # Perform feature selection
    train_selected, test_selected, selected_features = select_optimal_features(train_df, test_df)
    print(f"Selected {len(selected_features)} features")
    
    # Save the selected datasets
    train_selected.to_pickle(os.path.join(output_dir, "train_selected.pkl"))
    test_selected.to_pickle(os.path.join(output_dir, "test_selected.pkl"))
    print("Saved selected datasets")
    
    # Get feature importances using LightGBM
    X = train_selected.drop(columns=['TARGET'])
    y = train_selected['TARGET']
    
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Save and plot feature importances
    save_important_features(feature_importances, os.path.join(output_dir, "important_features.csv"))
    plot_feature_importances(feature_importances, output_file=os.path.join(output_dir, "feature_importances.png"))
    
    # Save selected feature names to text file
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print("Feature selection completed!")
if __name__ == "__main__":
    main()