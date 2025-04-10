import numpy as np
import pandas as pd
import gc
import time
import warnings
import os
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    """Context manager for timing code execution."""
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")

def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage of a dataframe by converting data types."""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


# Part 1: Process bureau and bureau_balance
def process_bureau_balance(bureau_balance_path, output_dir):
    """Process bureau_balance.csv and save aggregations."""
    with timer("Bureau Balance"):
        # Load bureau balance
        print("Loading bureau_balance.csv...")
        bureau_balance = pd.read_csv(bureau_balance_path)
        
        # Memory optimization
        bureau_balance = reduce_mem_usage(bureau_balance)
        
        # Basic aggregations
        print("Creating bureau_balance aggregations...")
        bb_aggregations = {
            'MONTHS_BALANCE': ['min', 'max', 'size']
        }
        
        # Process STATUS column to get counts of each status
        status_values = bureau_balance['STATUS'].value_counts().index.tolist()
        for status in status_values:
            bureau_balance[f'STATUS_{status}'] = bureau_balance['STATUS'].apply(lambda x: 1 if x == status else 0)
            bb_aggregations[f'STATUS_{status}'] = ['mean']
        
        # Perform aggregation
        bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bureau_balance_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance_agg.columns.tolist()])
        bureau_balance_agg.reset_index(inplace=True)
        
        # Save to disk
        print("Saving bureau_balance aggregations...")
        bureau_balance_agg.to_pickle(f"{output_dir}/bureau_balance_agg.pkl")
        
        # Free memory
        del bureau_balance
        gc.collect()
        
        return bureau_balance_agg

def process_bureau(bureau_path, bureau_balance_agg, output_dir):
    """Process bureau.csv and merge with bureau_balance aggregations."""
    with timer("Bureau"):
        # Load bureau
        print("Loading bureau.csv...")
        bureau = pd.read_csv(bureau_path)
        bureau = reduce_mem_usage(bureau)
        
        # Merge with bureau_balance_agg
        print("Merging with bureau_balance aggregations...")
        bureau = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')
        
        # Process categorical features
        cat_cols = [col for col in bureau.columns if bureau[col].dtype == 'object']
        for col in cat_cols:
            bureau[col] = bureau[col].fillna('Unknown')
            le = LabelEncoder()
            bureau[col] = le.fit_transform(bureau[col])
        
        # Create additional features
        print("Creating bureau features...")
        # Debt to credit ratio
        bureau['DEBT_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
        bureau['DEBT_CREDIT_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        bureau['DEBT_CREDIT_RATIO'].fillna(0, inplace=True)
        
        # Credit overdue to debt ratio
        bureau['CREDIT_OVERDUE_DEBT_RATIO'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / bureau['AMT_CREDIT_SUM_DEBT']
        bureau['CREDIT_OVERDUE_DEBT_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        bureau['CREDIT_OVERDUE_DEBT_RATIO'].fillna(0, inplace=True)
        
        # Define aggregations
        print("Aggregating bureau data...")
        num_cols = [c for c in bureau.columns if bureau[c].dtype != 'object' and c not in ['SK_ID_BUREAU', 'SK_ID_CURR']]
        agg_dict = {}
        
        # For most numeric columns, just take basic stats
        for col in num_cols:
            if col in ['DEBT_CREDIT_RATIO', 'CREDIT_OVERDUE_DEBT_RATIO']:
                # For critical ratio features, take more stats
                agg_dict[col] = ['min', 'mean', 'max', 'var']
            elif 'STATUS_' in col:
                # For status flags, just take mean (proportion)
                agg_dict[col] = ['mean']
            elif col in ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_MAX_OVERDUE']:
                # For important amount fields, take more stats
                agg_dict[col] = ['min', 'mean', 'max', 'sum']
            else:
                # For other fields, basic stats
                agg_dict[col] = ['mean']
        
        # Perform aggregation
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_dict)
        bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        
        # Count loans per client
        loans_count = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
        loans_count.columns = ['BUREAU_LOAN_COUNT']
        bureau_agg = bureau_agg.merge(loans_count, left_index=True, right_index=True, how='left')
        
        # Get most recent loan info
        recent_bureau = bureau.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'])
        recent_bureau = recent_bureau.groupby('SK_ID_CURR').last()
        recent_cols = ['DAYS_CREDIT', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT']
        recent_bureau = recent_bureau[recent_cols]
        recent_bureau.columns = ['BUREAU_RECENT_' + col for col in recent_cols]
        bureau_agg = bureau_agg.merge(recent_bureau, left_index=True, right_index=True, how='left')
        
        # Save to disk
        print("Saving bureau aggregations...")
        bureau_agg.to_pickle(f"{output_dir}/bureau_agg.pkl")
        
        # Free memory
        del bureau, bureau_balance_agg, loans_count, recent_bureau
        gc.collect()
        
        return bureau_agg

# Part 2: Process previous applications and related data
def process_previous_applications(prev_app_path, output_dir):
    """Process previous_application.csv and save basic aggregations."""
    with timer("Previous Applications Basic"):
        # Load previous applications
        print("Loading previous_application.csv...")
        prev_app = pd.read_csv(prev_app_path)
        prev_app = reduce_mem_usage(prev_app)
        
        # Process categorical features
        cat_cols = [col for col in prev_app.columns if prev_app[col].dtype == 'object']
        for col in cat_cols:
            prev_app[col] = prev_app[col].fillna('Unknown')
            le = LabelEncoder()
            prev_app[col] = le.fit_transform(prev_app[col])
        
        # Create additional features
        print("Creating previous application features...")
        # Credit to annuity ratio
        prev_app['CREDIT_TO_ANNUITY_RATIO'] = prev_app['AMT_CREDIT'] / prev_app['AMT_ANNUITY']
        prev_app['CREDIT_TO_ANNUITY_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        prev_app['CREDIT_TO_ANNUITY_RATIO'].fillna(0, inplace=True)
        
        # Credit to goods price ratio
        prev_app['CREDIT_TO_GOODS_RATIO'] = prev_app['AMT_CREDIT'] / prev_app['AMT_GOODS_PRICE']
        prev_app['CREDIT_TO_GOODS_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        prev_app['CREDIT_TO_GOODS_RATIO'].fillna(0, inplace=True)
        
        # Save SKIDs mapping for later use with related tables
        prev_app_ids = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].copy()
        prev_app_ids.to_pickle(f"{output_dir}/prev_app_ids.pkl")
        
        # Define aggregations
        print("Aggregating previous application data...")
        num_cols = [c for c in prev_app.columns if prev_app[c].dtype != 'object' and c not in ['SK_ID_PREV', 'SK_ID_CURR']]
        agg_dict = {}
        
        # For numeric columns, take basic stats
        for col in num_cols:
            if col in ['CREDIT_TO_ANNUITY_RATIO', 'CREDIT_TO_GOODS_RATIO', 'AMT_CREDIT', 'AMT_ANNUITY']:
                # For important features, take more stats
                agg_dict[col] = ['min', 'mean', 'max', 'sum']
            else:
                # For other fields, basic stats
                agg_dict[col] = ['mean']
        
        # Perform aggregation
        prev_app_agg = prev_app.groupby('SK_ID_CURR').agg(agg_dict)
        prev_app_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_app_agg.columns.tolist()])
        
        # Count applications per client
        app_count = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
        app_count.columns = ['PREV_APP_COUNT']
        prev_app_agg = prev_app_agg.merge(app_count, left_index=True, right_index=True, how='left')
        
        # Calculate approval rate
        approved = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 1]
        approved_count = approved.groupby('SK_ID_CURR').size()
        approval_rate = pd.Series(index=prev_app_agg.index, data=0)
        approval_rate[approved_count.index] = approved_count / prev_app_agg.loc[approved_count.index, 'PREV_APP_COUNT']
        prev_app_agg['PREV_APPROVAL_RATE'] = approval_rate
        
        # Save to disk
        print("Saving previous application aggregations...")
        prev_app_agg.to_pickle(f"{output_dir}/prev_app_agg.pkl")
        
        # Create subset for time-based features
        prev_app_time = prev_app[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_DECISION', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_CONTRACT_STATUS']]
        prev_app_time.to_pickle(f"{output_dir}/prev_app_time.pkl")
        
        # Free memory
        del prev_app, app_count, approved, approved_count, approval_rate
        gc.collect()
        
        return prev_app_agg, prev_app_ids

def process_pos_cash(pos_cash_path, prev_app_ids, output_dir):
    """Process POS_CASH_balance.csv and link with previous applications."""
    with timer("POS CASH Balance"):
        # Load POS CASH balance
        print("Loading POS_CASH_balance.csv...")
        pos = pd.read_csv(pos_cash_path)
        pos = reduce_mem_usage(pos)
        
        # Create additional features
        print("Creating POS CASH features...")
        # Installment ratio
        pos['INSTALMENT_FUTURE_RATIO'] = pos['CNT_INSTALMENT_FUTURE'] / pos['CNT_INSTALMENT']
        pos['INSTALMENT_FUTURE_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        pos['INSTALMENT_FUTURE_RATIO'].fillna(0, inplace=True)
        
        # Define basic aggregations
        print("Aggregating POS CASH data by SK_ID_PREV...")
        agg_dict = {
            'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
            'CNT_INSTALMENT': ['min', 'max', 'mean'],
            'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean'],
            'INSTALMENT_FUTURE_RATIO': ['mean']
        }
        
        # Aggregate by SK_ID_PREV
        pos_agg_by_prev = pos.groupby('SK_ID_PREV').agg(agg_dict)
        pos_agg_by_prev.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg_by_prev.columns.tolist()])
        pos_agg_by_prev.reset_index(inplace=True)
        
        # Link to SK_ID_CURR through previous applications
        print("Linking POS CASH data to clients...")
        pos_by_curr = prev_app_ids.merge(pos_agg_by_prev, on='SK_ID_PREV', how='left')
        
        # Aggregate by SK_ID_CURR
        pos_agg_cols = [col for col in pos_by_curr.columns if col != 'SK_ID_CURR' and col != 'SK_ID_PREV']
        pos_by_curr_agg = pos_by_curr.groupby('SK_ID_CURR').agg({col: ['mean', 'max'] for col in pos_agg_cols})
        pos_by_curr_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_by_curr_agg.columns.tolist()])
        
        # Save to disk
        print("Saving POS CASH aggregations...")
        pos_by_curr_agg.to_pickle(f"{output_dir}/pos_cash_agg.pkl")
        
        # Free memory
        del pos, pos_agg_by_prev, pos_by_curr
        gc.collect()
        
        return pos_by_curr_agg

def process_installments(installments_path, prev_app_ids, output_dir):
    """Process installments_payments.csv and link with previous applications."""
    with timer("Installments Payments"):
        # Load installments
        print("Loading installments_payments.csv...")
        ins = pd.read_csv(installments_path)
        ins = reduce_mem_usage(ins)
        
        # Create additional features
        print("Creating installment features...")
        # Payment ratio
        ins['PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        ins['PAYMENT_RATIO'].fillna(0, inplace=True)
        
        # Payment difference
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        
        # Days late
        ins['DAYS_LATE'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DAYS_LATE'] = ins['DAYS_LATE'].apply(lambda x: max(x, 0))
        
        # Late payment flag
        ins['IS_LATE'] = (ins['DAYS_LATE'] > 0).astype(int)
        
        # Define basic aggregations
        print("Aggregating installments data by SK_ID_PREV...")
        agg_dict = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
            'DAYS_INSTALMENT': ['min', 'max', 'mean'],
            'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
            'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'PAYMENT_RATIO': ['min', 'max', 'mean'],
            'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum'],
            'DAYS_LATE': ['max', 'mean', 'sum'],
            'IS_LATE': ['mean']
        }
        
        # Aggregate by SK_ID_PREV
        ins_agg_by_prev = ins.groupby('SK_ID_PREV').agg(agg_dict)
        ins_agg_by_prev.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg_by_prev.columns.tolist()])
        ins_agg_by_prev.reset_index(inplace=True)
        
        # Link to SK_ID_CURR through previous applications
        print("Linking installments data to clients...")
        ins_by_curr = prev_app_ids.merge(ins_agg_by_prev, on='SK_ID_PREV', how='left')
        
        # Aggregate by SK_ID_CURR
        ins_agg_cols = [col for col in ins_by_curr.columns if col != 'SK_ID_CURR' and col != 'SK_ID_PREV']
        ins_by_curr_agg = ins_by_curr.groupby('SK_ID_CURR').agg({col: ['mean', 'max'] for col in ins_agg_cols})
        ins_by_curr_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_by_curr_agg.columns.tolist()])
        
        # Save to disk
        print("Saving installments aggregations...")
        ins_by_curr_agg.to_pickle(f"{output_dir}/installments_agg.pkl")
        
        # Free memory
        del ins, ins_agg_by_prev, ins_by_curr
        gc.collect()
        
        return ins_by_curr_agg

def process_credit_card(credit_card_path, prev_app_ids, output_dir):
    """Process credit_card_balance.csv and link with previous applications."""
    with timer("Credit Card Balance"):
        # Load credit card balance
        print("Loading credit_card_balance.csv...")
        cc = pd.read_csv(credit_card_path)
        cc = reduce_mem_usage(cc)
        
        # Create additional features
        print("Creating credit card features...")
        # Balance to limit ratio
        cc['BALANCE_TO_LIMIT_RATIO'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
        cc['BALANCE_TO_LIMIT_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        cc['BALANCE_TO_LIMIT_RATIO'].fillna(0, inplace=True)
        
        # Drawing to limit ratio
        cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
        cc['DRAWING_LIMIT_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        cc['DRAWING_LIMIT_RATIO'].fillna(0, inplace=True)
        
        # Define basic aggregations
        print("Aggregating credit card data by SK_ID_PREV...")
        agg_dict = {
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            'AMT_BALANCE': ['max', 'mean', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
            'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT_CURRENT': ['max', 'mean', 'sum'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean'],
            'BALANCE_TO_LIMIT_RATIO': ['max', 'mean'],
            'DRAWING_LIMIT_RATIO': ['max', 'mean']
        }
        
        # Aggregate by SK_ID_PREV
        cc_agg_by_prev = cc.groupby('SK_ID_PREV').agg(agg_dict)
        cc_agg_by_prev.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg_by_prev.columns.tolist()])
        cc_agg_by_prev.reset_index(inplace=True)
        
        # Link to SK_ID_CURR through previous applications
        print("Linking credit card data to clients...")
        cc_by_curr = prev_app_ids.merge(cc_agg_by_prev, on='SK_ID_PREV', how='left')
        
        # Aggregate by SK_ID_CURR
        cc_agg_cols = [col for col in cc_by_curr.columns if col != 'SK_ID_CURR' and col != 'SK_ID_PREV']
        cc_by_curr_agg = cc_by_curr.groupby('SK_ID_CURR').agg({col: ['mean', 'max'] for col in cc_agg_cols})
        cc_by_curr_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_by_curr_agg.columns.tolist()])
        
        # Save to disk
        print("Saving credit card aggregations...")
        cc_by_curr_agg.to_pickle(f"{output_dir}/credit_card_agg.pkl")
        
        # Free memory
        del cc, cc_agg_by_prev, cc_by_curr
        gc.collect()
        
        return cc_by_curr_agg

# Part 3: Process application data and combine all features
def encode_categoricals(df):
    """One-hot encode categorical variables."""
    categorical_feats = [col for col in df.columns if df[col].dtype == 'object']
    if len(categorical_feats) == 0:
        return df, []
    
    df_copy = df.copy()
    for col in categorical_feats:
        df_copy[col] = df_copy[col].fillna('Unknown')
    
    # One-hot encode
    df_encoded = pd.get_dummies(df_copy, columns=categorical_feats, dummy_na=False)
    
    # Get newly created columns
    new_columns = [c for c in df_encoded.columns if c not in df.columns]
    
    return df_encoded, new_columns

def process_application_data(app_train_path, app_test_path, output_dir):
    """Process application data and create features."""
    with timer("Application Data"):
        # Load application data
        print("Loading application_train.csv and application_test.csv...")
        app_train = pd.read_csv(app_train_path)
        app_test = pd.read_csv(app_test_path)
        
        # Combine for preprocessing
        app_test['TARGET'] = np.nan
        df = pd.concat([app_train, app_test], axis=0)
        print(f"Combined application data shape: {df.shape}")
        
        # Memory optimization
        df = reduce_mem_usage(df)
        
        # Basic preprocessing
        print("Preprocessing application data...")
        
        # Remove columns with too many missing values
        missing_threshold = 0.75
        missing = df.isnull().sum() / len(df)
        missing_columns = missing[missing > missing_threshold].index.tolist()
        if missing_columns:
            print(f"Removing {len(missing_columns)} columns with >75% missing values")
            df.drop(missing_columns, axis=1, inplace=True)
        
        # Fill missing values for numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'TARGET']
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
            
        # Categorical columns will be handled later
        
        # Create basic features
        print("Creating application features...")
        
        # Age features
        df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
        df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
        df['DAYS_REGISTRATION'] = abs(df['DAYS_REGISTRATION'])
        df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])
        
        df['AGE_YEARS'] = df['DAYS_BIRTH'] / 365
        df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / 365
        df['REGISTRATION_YEARS'] = df['DAYS_REGISTRATION'] / 365
        df['ID_PUBLISH_YEARS'] = df['DAYS_ID_PUBLISH'] / 365
        
        # Flag 365243 days employed as missing
        df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == 365243
        df.loc[df['DAYS_EMPLOYED_ANOM'], 'DAYS_EMPLOYED'] = np.nan
        
        # Important ratios
        df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        
        # Replace infinite values
        for col in ['INCOME_CREDIT_RATIO', 'INCOME_PER_PERSON', 'ANNUITY_INCOME_RATIO', 'CREDIT_GOODS_RATIO']:
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            df[col].fillna(df[col].median(), inplace=True)
        
        # Save intermediate result to free memory later
        print("Saving processed application data...")
        df.to_pickle(f"{output_dir}/application_processed.pkl")
        
        # Save IDs for later merging
        app_ids = df[['SK_ID_CURR']].copy()
        app_ids.to_pickle(f"{output_dir}/app_ids.pkl")
        
        return df, app_ids

def merge_all_data(app_data, app_ids, output_dir):
    """Merge application data with all aggregated features."""
    with timer("Final Merging"):
        # Load all aggregated data
        print("Loading all aggregated features...")
        bureau_agg = pd.read_pickle(f"{output_dir}/bureau_agg.pkl")
        prev_app_agg = pd.read_pickle(f"{output_dir}/prev_app_agg.pkl")
        pos_agg = pd.read_pickle(f"{output_dir}/pos_cash_agg.pkl")
        ins_agg = pd.read_pickle(f"{output_dir}/installments_agg.pkl")
        cc_agg = pd.read_pickle(f"{output_dir}/credit_card_agg.pkl")
        
        # Prepare application data
        print("Encoding categorical features...")
        app_data, cat_cols = encode_categoricals(app_data)
        print(f"Created {len(cat_cols)} one-hot encoded features")
        
        # Start merging with manageable chunks
        print("Merging bureau features...")
        app_data_with_bureau = app_data.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        # Free memory
        del bureau_agg
        gc.collect()
        
        print("Merging previous application features...")
        app_data_with_bureau_prev = app_data_with_bureau.merge(prev_app_agg, on='SK_ID_CURR', how='left')
        
        # Free memory
        del app_data_with_bureau, prev_app_agg
        gc.collect()
        
        print("Merging POS CASH features...")
        app_data_with_bureau_prev_pos = app_data_with_bureau_prev.merge(pos_agg, on='SK_ID_CURR', how='left')
        
        # Free memory
        del app_data_with_bureau_prev, pos_agg
        gc.collect()
        
        print("Merging installments features...")
        app_data_with_bureau_prev_pos_ins = app_data_with_bureau_prev_pos.merge(ins_agg, on='SK_ID_CURR', how='left')
        
        # Free memory
        del app_data_with_bureau_prev_pos, ins_agg
        gc.collect()
        
        print("Merging credit card features...")
        final_data = app_data_with_bureau_prev_pos_ins.merge(cc_agg, on='SK_ID_CURR', how='left')
        
        # Free memory
        del app_data_with_bureau_prev_pos_ins, cc_agg
        gc.collect()
        
        # Fill NaN values
        print("Filling missing values...")
        for col in final_data.columns:
            if col != 'TARGET':
                final_data[col].fillna(0, inplace=True)
        
        # Split back to train and test
        print("Splitting back to train and test...")
        train_df = final_data[final_data['TARGET'].notna()].copy()
        test_df = final_data[final_data['TARGET'].isna()].copy()
        test_df.drop('TARGET', axis=1, inplace=True)
        
        print(f"Final train shape: {train_df.shape}, Final test shape: {test_df.shape}")
        
        # Save final datasets
        print("Saving final datasets...")
        train_df.to_pickle(f"{output_dir}/train_final.pkl")
        test_df.to_pickle(f"{output_dir}/test_final.pkl")
        
        # Also save as CSV (could be slower and use more memory)
        try:
            print("Attempting to save as CSV (this might cause memory issues)...")
            train_df.to_csv(f"{output_dir}/train_final.csv", index=False)
            test_df.to_csv(f"{output_dir}/test_final.csv", index=False)
        except Exception as e:
            print(f"Error saving as CSV: {e}")
            print("The data is still available in pickle format.")
        
        return train_df, test_df

def main():
    """Main pipeline for data processing with memory optimization."""
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    data_dir = os.path.join(project_root, "data/raw")
    output_dir = os.path.join(project_root, "data/processed")
    
    # Create output directory
    create_output_dir(output_dir)
    
    # Define file paths
    bureau_balance_path = f"{data_dir}/bureau_balance.csv"
    bureau_path = f"{data_dir}/bureau.csv"
    prev_app_path = f"{data_dir}/previous_application.csv"
    pos_cash_path = f"{data_dir}/POS_CASH_balance.csv"
    installments_path = f"{data_dir}/installments_payments.csv"
    credit_card_path = f"{data_dir}/credit_card_balance.csv"
    app_train_path = f"{data_dir}/application_train.csv"
    app_test_path = f"{data_dir}/application_test.csv"
    
    print("Starting memory-optimized data integration pipeline...")
    
    # Step 1: Process bureau data
    print("\n========== STEP 1: Bureau Data ==========")
    bureau_balance_agg = process_bureau_balance(bureau_balance_path, output_dir)
    bureau_agg = process_bureau(bureau_path, bureau_balance_agg, output_dir)
    
    # Free memory
    del bureau_balance_agg
    gc.collect()
    
    # Step 2: Process previous applications and related data
    print("\n========== STEP 2: Previous Applications ==========")
    prev_app_agg, prev_app_ids = process_previous_applications(prev_app_path, output_dir)
    
    # Process related tables one by one with smaller memory footprint
    print("\n========== STEP 3: Related Tables ==========")
    pos_agg = process_pos_cash(pos_cash_path, prev_app_ids, output_dir)
    ins_agg = process_installments(installments_path, prev_app_ids, output_dir)
    cc_agg = process_credit_card(credit_card_path, prev_app_ids, output_dir)
    
    # Free memory
    del prev_app_ids
    gc.collect()
    
    # Step 4: Process application data
    print("\n========== STEP 4: Application Data ==========")
    app_data, app_ids = process_application_data(app_train_path, app_test_path, output_dir)
    
    # Step 5: Merge everything together
    print("\n========== STEP 5: Final Merging ==========")
    train_df, test_df = merge_all_data(app_data, app_ids, output_dir)
    
    print("\nData integration pipeline completed successfully!")
    print(f"Final training dataset shape: {train_df.shape}")
    print(f"Final testing dataset shape: {test_df.shape}")
    print(f"All processed data saved to: {output_dir}")

if __name__ == "__main__":
    main()