# -*- coding: utf-8 -*-
"""
Created in 2025

Functionality: General utility functions

"""

import datetime
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings(action='ignore')
pd.set_option('display.encoding', 'gbk')


def getExponentialDecayWeight(halflife, period):
    '''
    Function Name: getExponentialDecayWeight
    Functionality: Obtain exponential decay weights, used for calculating multiple factor values.
    Input Parameters:
        halflife: The half-life period.
        period: The window period.
    Output:
        weightSeries: The computed weight series.
    '''
    alpha = np.power(0.5, 1 / halflife)
    weightSeries = np.arange(period)
    weightSeries = np.power(alpha, weightSeries)
    weightSeries = weightSeries / np.sum(weightSeries)
    weightSeries = weightSeries[::-1]
    return weightSeries


def gettickerlist(base):
    '''
    Function Name: gettickerlist
    Functionality: Retrieve all stock codes from the BASE data.
    Input Parameters:
        base: The BASE dataframe.
    Output:
        A list of ticker codes.
    '''
    tickerlist = []
    for i in base['S_INFO_WINDCODE'].unique():
        tickerlist.append(i)
    return tickerlist


def gettimelist(base):
    '''
    Function Name: gettimelist
    Functionality: Retrieve all dates from the BASE data.
    Input Parameters:
        base: The BASE dataframe.
    Output:
        A list of dates.
    '''
    timelist = []
    for i in base['TRADE_DT'].unique():
        timelist.append(i)
    return timelist


def IndustriesTimeSeries(base, IndustriesClass, AShareCalendar):
    '''
    Function Name: IndustriesTimeSeries
    Functionality: Convert industry change information into a time series on trading days.
                   This involves converting ENTRY_DT to the next trading day and then merging with BASE.
    Input Parameters:
        base: The BASE dataframe.
        IndustriesClass: DataFrame containing industry change information.
        AShareCalendar: Trading calendar data.
    Output:
        A DataFrame with complete time series industry information.
    '''
    IndustriesClass = IndustriesClass.sort_values(by=['S_INFO_WINDCODE', 'ENTRY_DT']).reset_index(drop=True)
    # Step 1: Convert ENTRY_DT in the industry change table to the next trading day (to improve efficiency)
    # Get the mapping between calendar dates and trading days
    dfDate = pd.DataFrame(pd.date_range('20000101', datetime.datetime.strftime(datetime.date.today(), '%Y%m%d'),
                                        freq='D', name='ENTRY_DT'))
    dfDate['ENTRY_DT'] = dfDate['ENTRY_DT'].apply(lambda x: x.strftime('%Y%m%d'))
    AShareCalendar.rename(columns={'TRADE_DAYS': 'TRADE_DT'}, inplace=True)
    dfDate = pd.merge(dfDate, AShareCalendar, left_on='ENTRY_DT', right_on='TRADE_DT', how='left')
    dfDate = dfDate.fillna(method='bfill')
    # ENTRY_DT_ALTER represents the next trading day for ENTRY_DT
    IndustriesClass = pd.merge(IndustriesClass, dfDate, on='ENTRY_DT', how='left')
    IndustriesClass.rename(columns={'TRADE_DT': 'ENTRY_DT_ALTER'}, inplace=True)
    # Next, convert the industry change information into a time series on trading days
    Result = pd.DataFrame(columns=['S_INFO_WINDCODE', 'TRADE_DT', 'INDUSTRIESNAME'])
    # Set the industry value based on ENTRY_DT_ALTER
    for stock in IndustriesClass['S_INFO_WINDCODE'].unique():
        IndustriesClassStock = IndustriesClass[IndustriesClass['S_INFO_WINDCODE'] == stock]
        for entry_dt_alter in IndustriesClassStock['ENTRY_DT_ALTER'].values:
            IndustriesClassStockPeriod = IndustriesClassStock[IndustriesClassStock['ENTRY_DT_ALTER'] == entry_dt_alter]
            Result.loc[len(Result)] = [stock, entry_dt_alter, IndustriesClassStockPeriod['INDUSTRIESNAME'].values[0]]
    
    # Merge with BASE to obtain complete time series industry information.
    # To preserve industry information for dates earlier than those in BASE (e.g., an entry_dt from 2003),
    # first perform an outer merge, then forward-fill NaNs, and finally merge with BASE.
    Result = pd.merge(Result, base, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='outer')
    Result.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], inplace=True)
    # Forward-fill NaN values
    Result = Result.groupby('S_INFO_WINDCODE').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
    Result = pd.merge(Result, base, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='right')
    return Result


def FreeFloatCap_Weight_df(BASE, EOD_DI):
    '''
    Function Name: FreeFloatCap_Weight_df
    Functionality: Obtain free float market capitalization and calculate weights for factor standardization.
    Input Parameters:
        BASE: The BASE dataframe.
        EOD_DI: The ASHAREEODDERIVATIVEINDICATOR table.
    Output:
        A DataFrame containing free float market capitalization weights.
    '''
    # Normalize free float market capitalization across the cross-section to obtain weights
    EOD_DI = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'FREE_FLOAT_CAP']]
    data = pd.merge(BASE, EOD_DI, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')
    data['WEIGHT'] = np.nan

    def func(group):
        group['WEIGHT'] = group['FREE_FLOAT_CAP'].dropna() / np.nansum(group['FREE_FLOAT_CAP'])  # np.nansum ignores NaNs when summing
        return group

    data = data.groupby('TRADE_DT').apply(func).reset_index(drop=True)
    return data[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]


def FreeFloatCap_Weight_matrix(EOD_DI):
    '''
    Function Name: FreeFloatCap_Weight_matrix
    Functionality: Obtain free float market capitalization and calculate weights for factor standardization in matrix format.
    Input Parameters:
        EOD_DI: The ASHAREEODDERIVATIVEINDICATOR table.
    Output:
        weight: Free float market capitalization weights in matrix format.
    '''
    # Normalize free float market capitalization across the cross-section to obtain weights
    EOD_DI = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'FREE_FLOAT_CAP']]
    matrix_FreeCAP = pd.pivot(EOD_DI, index='S_INFO_WINDCODE', values='FREE_FLOAT_CAP', columns='TRADE_DT')
    weight = pd.DataFrame(index=matrix_FreeCAP.index, columns=matrix_FreeCAP.columns)
    n = len(matrix_FreeCAP.columns)
    for i in tqdm(range(n)):
        data = matrix_FreeCAP.iloc[:, i].dropna() / np.nansum(matrix_FreeCAP.iloc[:, i])  # np.nansum ignores NaNs when summing
        weight.iloc[:, i][data.index] = data.values
    return weight


def process_factor(factor, weight_df, I, mad):
    '''
    Function Name: process_factor
    Functionality: Preprocess factor values by performing optional outlier removal, standardization, and filling of missing values.
    Input Parameters:
        factor: The factor values.
        weight_df: The standardization weights.
        I: Stock industry information.
        mad: Boolean flag indicating whether to perform outlier removal (using MAD).
    Output:
        factor_fillna: Factor values after filling missing data.
    '''
    factor = factor.stack().reset_index()
    factor = factor.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    factor = pd.merge(weight_df, factor, how='left')
    if mad:
        # Remove outliers and standardize
        factor_stand = standardize_mad_factor(factor)
    else:
        # Standardize only without outlier removal
        factor_stand = standardize_factor(factor)
    # Fill missing values (using industry median)
    factor_fillna = fillna_factor(factor_stand, I)
    return factor_fillna


def standardize_factor(factor):
    '''
    Function Name: standardize_factor
    Functionality: Standardize factor values (for some factors that do not require outlier removal).
    Input Parameters:
        factor: The factor values.
        (Note: weight_df is embedded in the factor dataframe here.)
    Output:
        The standardized factor values.
    '''
    # Define standardized factor values
    factor['FACTOR_stand'] = np.nan

    def func(group):
        # Standardization
        fa = group['FACTOR']
        weight = group['WEIGHT']
        if np.nanstd(fa.dropna()) == 0:
            group['FACTOR_stand'] = group['FACTOR']
            return group
        stand_fa = (group['FACTOR'].dropna() - np.nansum(weight.dropna() * group['FACTOR'].dropna())) / np.nanstd(group['FACTOR'].dropna())
        group['FACTOR_stand'] = stand_fa
        return group

    factor = factor.groupby('TRADE_DT').apply(func).reset_index(drop=True)
    factor = factor[['S_INFO_WINDCODE', 'TRADE_DT', 'FACTOR_stand']]
    factor.rename(columns={'FACTOR_stand': 'FACTOR'}, inplace=True)
    return factor


def mad(fa, n=5):
    '''
    Function Name: mad
    Functionality: Median-based outlier removal function.
                   The input is a Series (indexed by stock code) with factor values.
                   'n' is the multiplier for the deviation threshold.
    Input Parameters:
        fa: The factor values as a Series.
        n: The threshold multiplier.
    Output:
        A Series with extreme values clipped.
    '''
    r = fa.dropna().copy()
    median = r.quantile(0.5)
    new_median = abs(r - median).quantile(0.5)
    up = median + n * new_median
    down = median - n * new_median
    return np.clip(r, down, up)


def standardize_mad_factor(factor):
    '''
    Function Name: standardize_mad_factor
    Functionality: Remove outliers and standardize factor values.
    Input Parameters:
        factor: The factor values.
        (Note: weight_df is embedded in the factor dataframe here.)
    Output:
        The factor values after outlier removal and standardization.
    '''
    # Define columns for outlier-removed and standardized factor values
    factor['FACTOR_mad'] = np.nan
    factor['FACTOR_mad_stand'] = np.nan
    # Remove outliers and standardize
    def func(group):
        # Outlier removal
        fa = group['FACTOR']
        # If the entire group is NaN, skip processing
        if len(fa.dropna()) == 0:
            return group
        mad_fa = mad(fa, n=5)
        group['FACTOR_mad'] = mad_fa
        # Standardization
        weight = group['WEIGHT']
        if np.nanstd(mad_fa.dropna()) == 0:
            group['FACTOR_mad_stand'] = group['FACTOR_mad']
            return group
        stand_fa = (group['FACTOR_mad'].dropna() - np.nansum(weight.dropna() * group['FACTOR_mad'].dropna())) / np.nanstd(group['FACTOR_mad'].dropna())
        group['FACTOR_mad_stand'] = stand_fa
        return group

    factor = factor.groupby('TRADE_DT').apply(func).reset_index(drop=True)
    factor = factor[['S_INFO_WINDCODE', 'TRADE_DT', 'FACTOR_mad_stand']]
    factor.rename(columns={'FACTOR_mad_stand': 'FACTOR'}, inplace=True)
    return factor


def fillna_factor(factor, I):
    '''
    Function Name: fillna_factor
    Functionality: Fill missing factor values using the median value within each industry.
    Input Parameters:
        factor: The factor values.
        I: Stock industry information.
    Output:
        factor_fillna: Factor values after missing data have been filled.
    '''
    factor_df = pd.merge(factor, I, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')

    def fill(df):
        # List of industry names (as given in Chinese)
        industry_names = ['交通运输', '传媒', '农林牧渔', '医药', '商贸零售', '国防军工',
                          '基础化工', '家电', '建材', '建筑', '房地产', '有色金属', '机械', '汽车', '消费者服务', '煤炭',
                          '电力及公用事业', '电力设备及新能源', '电子', '石油石化', '纺织服装', '综合', '综合金融', '计算机',
                          '轻工制造', '通信', '钢铁', '银行', '非银行金融', '食品饮料']
        for ind_name in industry_names:
            df.loc[df['INDUSTRIESNAME'] == ind_name, 'FACTOR'] = df.loc[df['INDUSTRIESNAME'] == ind_name, 'FACTOR'].fillna(
                df.loc[df['INDUSTRIESNAME'] == ind_name, 'FACTOR'].median(axis=0))
        return df

    tqdm.pandas(desc="Filling missing values")
    factor_fillna_df = factor_df.groupby('TRADE_DT').progress_apply(fill).reset_index(drop=True)
    return factor_fillna_df[['S_INFO_WINDCODE', 'TRADE_DT', 'FACTOR']]


def getAllPeriod(INCOME):
    '''
    Function Name: getAllPeriod
    Functionality: Retrieve all financial report announcement dates for stocks from the AShareIncome table.
    Input Parameters:
        INCOME: The AShareIncome table.
    Output:
        AllPeriod: Financial report announcement dates.
    '''
    AllPeriod = INCOME[['S_INFO_WINDCODE', 'REPORT_PERIOD', 'ANN_DT_ALTER']]
    AllPeriod = AllPeriod.drop_duplicates(subset=['S_INFO_WINDCODE', 'REPORT_PERIOD', 'ANN_DT_ALTER'],
                                           keep='first').reset_index(drop=True)
    return AllPeriod


def Process_ANN_DT(df, AShareCalendar):
    '''
    Function Name: Process_ANN_DT
    Functionality: Move ANN_DT on non-trading days to the next trading day.
    Input Parameters:
        df: Data to be processed.
        AShareCalendar: Trading calendar table.
    Output:
        df: Processed data with adjusted ANN_DT.
    '''
    # Obtain all trading and calendar dates; align them using backfill
    dfDate = pd.DataFrame(pd.date_range('20000101', datetime.datetime.strftime(datetime.date.today(), '%Y%m%d'),
                                        freq='D', name='NATURE_DT'))
    dfDate['NATURE_DT'] = dfDate['NATURE_DT'].apply(lambda x: x.strftime('%Y%m%d'))
    dfDate = dfDate.merge(AShareCalendar, left_on='NATURE_DT', right_on='TRADE_DAYS', how='left')
    dfDate = dfDate.fillna(method='bfill')
    # Adjust the original ANN_DT in df
    df = df.merge(dfDate, left_on='ANN_DT', right_on='NATURE_DT', how='left')
    df.rename(columns={'TRADE_DAYS': 'ANN_DT_ALTER'}, inplace=True)
    df.drop(labels=['NATURE_DT'], axis=1, inplace=True)
    return df


def Process_Stock_Code(df, tickerlist):
    '''
    Function Name: Process_Stock_Code
    Functionality: Convert stock code format from "000001" to "000001.SZ".
    Input Parameters:
        df: Data to be processed.
        tickerlist: List of all stock codes (with suffix format).
    Output:
        df: Processed data with adjusted stock code format.
    '''
    dfStockCode = pd.DataFrame(tickerlist, columns=['S_INFO_WINDCODE_LONG'])
    dfStockCode['S_INFO_WINDCODE'] = dfStockCode['S_INFO_WINDCODE_LONG'].apply(lambda x: x[:6])
    # Adjust the original S_INFO_WINDCODE in df
    df = df.merge(dfStockCode, on='S_INFO_WINDCODE', how='left')
    df.rename(columns={'S_INFO_WINDCODE': 'S_INFO_WINDCODE_short'}, inplace=True)
    df.rename(columns={'S_INFO_WINDCODE_LONG': 'S_INFO_WINDCODE'}, inplace=True)
    return df


def next_month(date):
    '''
    Function Name: next_month
    Functionality: Given a date in "yyyymm" format, output the next month.
    Input Parameters:
        date: A date in "yyyymm" format.
    Output:
        The next month's date in "yyyymm" format.
    '''
    year = date[0:4]
    month = date[4:6]
    if month < '09':
        return year + '0' + str(int(month) + 1)
    elif month < '12':
        return year + str(int(month) + 1)
    else:
        return str(int(year) + 1) + '01'


def Align(data_matrix, BASE):
    '''
    Function Name: Align
    Functionality: Final step for factor generation. Align the matrix of factor values with BASE using trading days and stock codes.
    Input Parameters:
        data_matrix: Factor values in matrix format.
        BASE: Panel format BASE dataframe.
    Output:
        result_matrix: Aligned factor values in matrix format.
    '''
    data_stack = data_matrix.stack().reset_index()
    data_stack.columns = ['S_INFO_WINDCODE', 'TRADE_DT', 'FACTOR']
    # Merge with all trading days and stock codes, then sort
    data_stack = pd.merge(data_stack, BASE, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='right')
    data_stack.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], inplace=True)
    # Forward-fill NaN values
    data_stack = data_stack.groupby('S_INFO_WINDCODE').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
    # Pivot to matrix form
    result_matrix = pd.pivot(data_stack, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    return result_matrix


def factor_format_adjust(factor, name):
    '''
    Function Name: factor_format_adjust
    Functionality: Convert factor format from matrix to panel.
    Input Parameters:
        factor: Factor values in matrix format.
        name: Name of the factor.
    Output:
        Factor: Factor values in panel format.
    '''
    factor = factor.rename_axis('S_INFO_WINDCODE')
    factor.columns = factor.columns.astype(int)
    Factor = pd.melt(factor.reset_index(), id_vars='S_INFO_WINDCODE')
    col_new = ['S_INFO_WINDCODE', 'TRADE_DT', name]
    Factor.columns = col_new
    Factor['TRADE_DT'] = Factor['TRADE_DT'].astype(str)
    return Factor


def next_date(data):
    '''
    Function Name: next_date
    Functionality: Align factor exposure at period t with return at period t+1.
    Input Parameters:
        data: The input dataframe.
    Output:
        data: The dataframe with the "T" column shifted to represent the next period.
    '''
    data['T'] = data['T-1'].shift(-1)
    return data


def next_rtn(data):
    '''
    Function Name: next_rtn
    Functionality: Calculate the next period return for factor regression.
    Input Parameters:
        data: The input dataframe.
    Output:
        data: The dataframe with a new column "next_Rtn" representing the next period return.
    '''
    data['next_Rtn'] = data['S_DQ_ADJCLOSE'].shift(-1) / data['S_DQ_ADJCLOSE'] - 1
    return data


def trading(data):
    '''
    Function Name: trading
    Functionality: Determine whether trading is possible in the next period for factor regression.
    Input Parameters:
        data: The input dataframe.
    Output:
        data: The dataframe with a new column "trading" indicating if trading volume is positive.
    '''
    data['trading'] = data['S_DQ_VOLUME'] > 0
    return data
