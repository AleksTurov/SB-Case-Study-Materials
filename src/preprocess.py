import pandas as pd

def preprocess_data_rents(df):
    """
    Data preprocessing: handling missing data, encoding categorical variables, and scaling numerical features.
    
    :param df: DataFrame with raw data.
    :return: DataFrame with preprocessed data.
    """

    # Convert date columns to datetime
    df[['registration_date', 'contract_start_date', 'contract_end_date', 'req_from', 'req_to']] = df[['registration_date', 'contract_start_date', 'contract_end_date', 'req_from', 'req_to']].apply(pd.to_datetime)
    
    # Replace 't'/'f' with True/False
    df['is_freehold'] = df['is_freehold'].replace({'t': True, 'f': False})
    
    # Calculate time deltas
    df['delta_time_registration'] = (df['registration_date'] - df['contract_start_date']).dt.days
    df['delta_time_reg_from'] = (df['registration_date'] - df['req_from']).dt.days
    df['time_reg'] = (df['req_to'] - df['req_from']).dt.days
    df['time_contract'] = (df['contract_end_date'] - df['contract_start_date']).dt.days

    # Sort by version number and drop duplicates
    df.sort_values(['contract_amount', 'version_number','registration_date'], ascending=False, inplace=True)
    
    # Replace NaN values in categorical features with 'missing'
    df = df.apply(lambda x: x.fillna('missing') if x.dtype == 'object' else x)
    
    # Convert categorical features to numerical codes
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        df[col] = df[col].astype('category').cat.codes
        
    return df

def preprocess_data_trans(df):
    """
    Data preprocessing: handling missing data, encoding categorical variables, and scaling numerical features.
    
    :param df: DataFrame with raw data.
    :return: DataFrame with preprocessed data.
    """

    df['is_offplan'] = df['is_offplan'].replace({'t': True, 'f': False})
    df['is_freehold'] = df['is_freehold_text'].replace({'Free Hold': True, 'Non Free Hold': False})
    
    # Calculate time deltas
    df[['req_to', 'transaction_datetime', 'req_from']] = df[['transaction_datetime', 'req_from', 'req_to']].apply(pd.to_datetime)
    df['delta_time_reg_to'] = (df['req_to'] - df['transaction_datetime']).dt.days
    df['delta_time_reg_from'] = (df['req_from'] - df['transaction_datetime']).dt.days
    df['time_reg'] = (df['req_from'] - df['req_to']).dt.days

    # Sort by version number and drop duplicates
    df.sort_values(['transaction_datetime', 'amount'], ascending=False, inplace=True)
    
    # Replace NaN values in categorical features with 'missing'
    df = df.apply(lambda x: x.fillna('missing') if x.dtype == 'object' else x)
    
    # Convert categorical features to numerical codes
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        df[col] = df[col].astype('category').cat.codes
        
    return df