# -*- coding: utf-8 -*-
"""
Created in 2025

Functionality: Factor generation program

"""

import numpy.matlib
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from functools import reduce
import statsmodels.api as sm
from Tools import *
from LoadSQL import *
import sklearn.linear_model as LM
import warnings
warnings.filterwarnings(action='ignore')
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
pd.set_option('display.encoding', 'gbk')


# ---------------------------
# Factor Generation Functions
# ---------------------------

def LNCAP(BASE, EOD_DI):
    """
    Category: Size/Size/
    Factor Number: 1
    Factor Name: LNCAP: Log of Market Capitalization
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    LNCAP = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_VAL_MV']]
    # Calculate the natural logarithm of market capitalization (S_VAL_MV is in 10,000 Yuan)
    LNCAP['LNCAP'] = np.log(10000 * LNCAP['S_VAL_MV'])
    LNCAP = LNCAP[['S_INFO_WINDCODE', 'TRADE_DT', 'LNCAP']]
    matrix_LNCAP = pd.pivot(LNCAP, index='S_INFO_WINDCODE', values='LNCAP', columns='TRADE_DT')
    # Align with the base data
    Result = Align(matrix_LNCAP, BASE)
    return Result


def MIDCAP(BASE, EOD_DI):
    """
    Category: Size/Mid Capitalization/
    Factor Number: 2
    Factor Name: MIDCAP: Cube of Size Exposure
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    MIDCAP = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_VAL_MV', 'FREE_FLOAT_CAP']]
    # Calculate the natural logarithm of market capitalization (S_VAL_MV is in 10,000 Yuan)
    MIDCAP['LNCAP'] = np.log(10000 * MIDCAP['S_VAL_MV'])
    MIDCAP = MIDCAP[['S_INFO_WINDCODE', 'TRADE_DT', 'LNCAP', 'FREE_FLOAT_CAP']]
    # Replace inf values with NaN
    MIDCAP.replace([np.inf, -np.inf], np.nan, inplace=True)
    MIDCAP.dropna(axis=0, how='any', inplace=True)
    MIDCAP['X'] = np.nan
    # Free float market cap weighted standardization
    def normalize_LNCAP(df):
        MV = df['FREE_FLOAT_CAP']
        LNCAP = df['LNCAP']
        WI = MV / np.nansum(MV)
        LNCAP_mean = np.nansum(LNCAP * WI)
        LNCAP_std = np.nanstd(LNCAP)
        LNCAP_standardized = (LNCAP - LNCAP_mean) / LNCAP_std
        df['X'] = LNCAP_standardized
        return df
    MIDCAP = MIDCAP.groupby('TRADE_DT').apply(normalize_LNCAP)
    MIDCAP.reset_index(drop=True, inplace=True)
    MIDCAP['MIDCAP'] = np.nan
    MIDCAP['X_3'] = np.power(MIDCAP['X'], 3)
    MIDCAP = MIDCAP[['S_INFO_WINDCODE', 'TRADE_DT', 'X', 'X_3', 'MIDCAP']]
    # Regress X^3 on X and take the residual as MIDCAP (cross-sectional regression)
    def calculate_residuals(df):
        X_vals = df['X'].values.reshape((-1, 1))
        Y_vals = df['X_3']
        modelLR = LM.LinearRegression(fit_intercept=False)
        modelLR.fit(X_vals, Y_vals)
        y_pred = modelLR.predict(X_vals)
        residual = Y_vals - y_pred
        df['MIDCAP'] = residual
        return df
    MIDCAP = MIDCAP.groupby('TRADE_DT').apply(calculate_residuals)
    MIDCAP = MIDCAP[['S_INFO_WINDCODE', 'TRADE_DT', 'MIDCAP']]
    # For pandas 2.0, reset index before pivot_table
    MIDCAP.reset_index(drop=True, inplace=True)
    matrix_MIDCAP = pd.pivot_table(MIDCAP, index='S_INFO_WINDCODE', values='MIDCAP', columns='TRADE_DT')
    # Align with the base data
    Result = Align(matrix_MIDCAP, BASE)
    return Result


def HBETA(BASE, EOD, INDEX_EOD):
    """
    Category: Volatility/Beta/
    Factor Number: 3
    Factor Name: HBETA: Historical Beta
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
        INDEX_EOD: Data from the AindexEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    # For pandas 2.0, reset index before pivot_table
    RTN.reset_index(drop=True, inplace=True)
    matrix_RTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    # Calculate returns for the benchmark index (China Securities 500 Index)
    MKT = INDEX_EOD[INDEX_EOD['S_INFO_WINDCODE'] == '000985.SH']
    MKT['Rtn'] = MKT['S_DQ_CLOSE'] / MKT['S_DQ_PRECLOSE'] - 1
    MKT.reset_index(drop=True, inplace=True)
    matrix_MKT = pd.pivot_table(MKT, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    matrix_BETA = pd.DataFrame(index=matrix_RTN.index, columns=matrix_RTN.columns)
    # m: number of stocks, n: number of time points
    m = len(matrix_BETA.index)
    n = len(matrix_BETA.columns)
    period = 504
    halflife = 252
    weightSeries = getExponentialDecayWeight(halflife, period)
    # For each stock, regress its return series on the benchmark returns using exponentially decaying weights
    for j in tqdm(range(n - period + 1)):
        X = matrix_MKT.iloc[:, j:j+period].T
        X = sm.add_constant(X)
        y = matrix_RTN.iloc[:, j:j+period]
        B = sm.WLS(y.T, X, weights=weightSeries.T).fit()
        matrix_BETA.iloc[:, j + period - 1] = B.params.values[1]
    # Align with the base data
    Result = Align(matrix_BETA, BASE)
    return Result


def HSIGMA(BASE, EOD, INDEX_EOD):
    """
    Category: Volatility/Residual Volatility/
    Factor Number: 4
    Factor Name: HSIGMA: Historical Sigma
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
        INDEX_EOD: Data from the AindexEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN.reset_index(drop=True, inplace=True)
    matrix_RTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    # Returns for the benchmark index (China Securities 500 Index)
    MKT = INDEX_EOD[INDEX_EOD['S_INFO_WINDCODE'] == '000985.SH']
    MKT['Rtn'] = MKT['S_DQ_CLOSE'] / MKT['S_DQ_PRECLOSE'] - 1
    MKT.reset_index(drop=True, inplace=True)
    matrix_MKT = pd.pivot(MKT, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    matrix_HSIGMA = pd.DataFrame(index=matrix_RTN.index, columns=matrix_RTN.columns)
    m = len(matrix_HSIGMA.index)
    n = len(matrix_HSIGMA.columns)
    period = 504
    halflife = 252
    weightSeries = getExponentialDecayWeight(halflife, period)
    # For each stock, run a weighted regression of its return series on the benchmark returns
    for j in tqdm(range(n - period + 1)):
        X = matrix_MKT.iloc[:, j:j+period]
        X = sm.add_constant(X)
        y = matrix_RTN.iloc[:, j:j+period]
        B = sm.WLS(y.T, X.T, weights=weightSeries.T).fit()
        matrix_HSIGMA.iloc[:, j + period - 1] = np.nanstd(B.resid.T, axis=1)
    # Align with the base data
    Result = Align(matrix_HSIGMA, BASE)
    return Result


def DASTD(BASE, EOD):
    """
    Category: Volatility/Residual Volatility/
    Factor Number: 5
    Factor Name: DASTD: Daily Standard Deviation
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN.reset_index(drop=True, inplace=True)
    matrix_RTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    matrix_DASTD = pd.DataFrame(index=matrix_RTN.index, columns=matrix_RTN.columns)
    m = len(matrix_DASTD.index)
    n = len(matrix_DASTD.columns)
    # period: number of trading days; halflife: half-life in days
    period = 252
    halflife = 42
    weightSeries = getExponentialDecayWeight(halflife, period)
    V2 = np.sum(np.power(weightSeries, 2))
    weightSeries = np.matlib.repmat(weightSeries, m, 1)
    for i in tqdm(range(n - period + 1)):
        maP = np.nansum(matrix_RTN.iloc[:, i:i+period] * weightSeries, axis=1)
        maP = np.matlib.repmat(maP, period, 1).T
        matrix_DASTD.iloc[:, i + period - 1] = (1 / (1 - V2)) * np.nansum(weightSeries * np.power(matrix_RTN.iloc[:, i:i+period] - maP, 2), axis=1)
    matrix_DASTD = np.sqrt(matrix_DASTD.astype('float64'))
    # Replace all-zero factor series (resulting from summing all NaNs) with NaN
    matrix_DASTD = matrix_DASTD.replace(0, np.nan)
    # Align with the base data
    Result = Align(matrix_DASTD, BASE)
    return Result


def CMRA(BASE, EOD):
    """
    Category: Volatility/Residual Volatility/
    Factor Number: 6
    Factor Name: CMRA: Cumulative Range
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    # Calculate log returns
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN['ln(1+Rtn)'] = np.log(RTN['Rtn'] + 1)
    RTN.reset_index(drop=True, inplace=True)
    matrix_LNRTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='ln(1+Rtn)', columns='TRADE_DT')
    matrix_CMRA = pd.DataFrame(index=matrix_LNRTN.index, columns=matrix_LNRTN.columns)
    m = len(matrix_CMRA.index)
    n = len(matrix_CMRA.columns)
    month = list(range(1, 13))
    temp = pd.DataFrame(index=matrix_LNRTN.index, columns=month)
    Zmax = pd.DataFrame(index=matrix_LNRTN.index, columns=matrix_LNRTN.columns)
    Zmin = pd.DataFrame(index=matrix_LNRTN.index, columns=matrix_LNRTN.columns)
    period = 252
    for i in tqdm(range(n - period + 1)):
        for j in range(12):
            temp.iloc[:, j] = np.nansum(matrix_LNRTN.iloc[:, i + 21*(12 - j - 1): i + 252], axis=1)
        Zmax.iloc[:, i + period - 1] = np.max(temp, axis=1)
        Zmin.iloc[:, i + period - 1] = np.min(temp, axis=1)
        matrix_CMRA.iloc[:, i + period - 1] = Zmax.iloc[:, i + period - 1] - Zmin.iloc[:, i + period - 1]
    # Replace factor series that sum to 0 (i.e. no trading data) with NaN
    matrix_CMRA = matrix_CMRA.replace(0, np.nan)
    # Align with the base data
    Result = Align(matrix_CMRA, BASE)
    return Result


def STOM(BASE, EOD_DI):
    """
    Category: Liquidity/Liquidity/
    Factor Number: 7
    Factor Name: STOM: Monthly Share Turnover
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    # Obtain turnover data
    Turnover = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_FREETURNOVER']]
    # Replace 0 values (indicating no trading) with NaN
    Turnover.loc[Turnover['S_DQ_FREETURNOVER'] == 0, 'S_DQ_FREETURNOVER'] = np.nan
    # Reset index for pivoting
    Turnover.reset_index(drop=True, inplace=True)
    matrix_Turnover = pd.pivot(Turnover, index='S_INFO_WINDCODE', values='S_DQ_FREETURNOVER', columns='TRADE_DT')
    matrix_STOM = pd.DataFrame(index=matrix_Turnover.index, columns=matrix_Turnover.columns)
    period = 21
    m = len(matrix_STOM.index)
    n = len(matrix_STOM.columns)
    # Calculate STOM: sum turnover over a 21-day window
    for i in tqdm(range(n - period + 1)):
        matrix_STOM.iloc[:, i + period - 1] = np.nansum(matrix_Turnover.iloc[:, i:i+period], axis=1)
    # Replace zero sums with NaN
    matrix_STOM = matrix_STOM.replace(0, np.nan)
    # Take the logarithm of the turnover sum
    matrix_STOM = matrix_STOM.applymap(lambda x: np.log(x) if isinstance(x, (int, float)) else x)
    # Align with the base data
    Result = Align(matrix_STOM, BASE)
    return Result


def STOQ(BASE, EOD_DI):
    """
    Category: Liquidity/Liquidity/
    Factor Number: 8
    Factor Name: STOQ: Quarterly Share Turnover
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    # Obtain turnover data
    Turnover = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_FREETURNOVER']]
    # Replace 0 values with NaN
    Turnover.loc[Turnover['S_DQ_FREETURNOVER'] == 0, 'S_DQ_FREETURNOVER'] = np.nan
    Turnover.reset_index(drop=True, inplace=True)
    matrix_Turnover = pd.pivot(Turnover, index='S_INFO_WINDCODE', values='S_DQ_FREETURNOVER', columns='TRADE_DT')
    matrix_STOQ = pd.DataFrame(index=matrix_Turnover.index, columns=matrix_Turnover.columns)
    period = 63
    m = len(matrix_STOQ.index)
    n = len(matrix_STOQ.columns)
    for i in tqdm(range(n - period + 1)):
        matrix_STOQ.iloc[:, i + period - 1] = np.nansum(matrix_Turnover.iloc[:, i:i+period], axis=1) / 3
    matrix_STOQ = matrix_STOQ.replace(0, np.nan)
    # Take the logarithm
    matrix_STOQ = matrix_STOQ.applymap(lambda x: np.log(x) if isinstance(x, (int, float)) else x)
    Result = Align(matrix_STOQ, BASE)
    return Result


def STOA(BASE, EOD_DI):
    """
    Category: Liquidity/Liquidity/
    Factor Number: 9
    Factor Name: STOA: Annual Share Turnover
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    Turnover = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_FREETURNOVER']]
    Turnover.loc[Turnover['S_DQ_FREETURNOVER'] == 0, 'S_DQ_FREETURNOVER'] = np.nan
    Turnover.reset_index(drop=True, inplace=True)
    matrix_Turnover = pd.pivot(Turnover, index='S_INFO_WINDCODE', values='S_DQ_FREETURNOVER', columns='TRADE_DT')
    matrix_STOA = pd.DataFrame(index=matrix_Turnover.index, columns=matrix_Turnover.columns)
    period = 252
    m = len(matrix_STOA.index)
    n = len(matrix_STOA.columns)
    # Calculate STOA: sum turnover over 252 trading days divided by 12
    for i in tqdm(range(n - period + 1)):
        matrix_STOA.iloc[:, i + period - 1] = np.nansum(matrix_Turnover.iloc[:, i:i+period], axis=1) / 12
    matrix_STOA = matrix_STOA.replace(0, np.nan)
    matrix_STOA = np.log(matrix_STOA.astype('float'))
    Result = Align(matrix_STOA, BASE)
    return Result


def ATVR(BASE, EOD_DI):
    """
    Category: Liquidity/Liquidity/
    Factor Number: 10
    Factor Name: ATVR: Annualized Traded Value Ratio
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    # Extract turnover data
    Turnover = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_FREETURNOVER']]
    Turnover.loc[Turnover['S_DQ_FREETURNOVER'] == 0, 'S_DQ_FREETURNOVER'] = np.nan
    matrix_Turnover = pd.pivot(Turnover, index='S_INFO_WINDCODE', values='S_DQ_FREETURNOVER', columns='TRADE_DT')
    matrix_ATVR = pd.DataFrame(index=matrix_Turnover.index, columns=matrix_Turnover.columns)
    m = len(matrix_ATVR.index)  # Number of stocks
    n = len(matrix_ATVR.columns)  # Number of time points
    period = 252  # Window length
    halflife = 63  # Half-life
    weightSeries = getExponentialDecayWeight(halflife, period)
    weightSeries = np.matlib.repmat(weightSeries, m, 1)
    for i in tqdm(range(n - period + 1)):
        matrix_ATVR.iloc[:, i + period - 1] = np.nansum(matrix_Turnover.iloc[:, i:i+period] * weightSeries, axis=1)
    matrix_ATVR = matrix_ATVR.replace(0, np.nan)
    Result = Align(matrix_ATVR, BASE)
    return Result


def STREV(BASE, EOD):
    """
    Category: Momentum/Short-Term Reversal/
    Factor Number: 11
    Factor Name: STREV: Short-term Reversal
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    # For short-term reversal, compute the weighted sum of log returns over a one-month period.
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN['ln(1+Rtn)'] = np.log(RTN['Rtn'] + 1)
    matrix_lnRTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='ln(1+Rtn)', columns='TRADE_DT')
    matrix_STR = pd.DataFrame(index=matrix_lnRTN.index, columns=matrix_lnRTN.columns)
    m = len(matrix_STR.index)
    n = len(matrix_STR.columns)
    period = 63  # Window length
    halflife = 10  # Half-life
    weightSeries = getExponentialDecayWeight(halflife, period)
    weightSeries = np.matlib.repmat(weightSeries, m, 1)
    for i in tqdm(range(n - period + 1)):
        matrix_STR.iloc[:, i + period - 1] = np.nansum(matrix_lnRTN.iloc[:, i:i+period] * weightSeries, axis=1)
    matrix_STR = matrix_STR.replace(0, np.nan)
    # Then, lag by 1 day and perform 3-day smoothing
    period_2 = 3
    lag_2 = 1
    matrix_STR_lag = pd.DataFrame(index=matrix_STR.index, columns=matrix_STR.columns)
    for i in tqdm(range(n - period_2 - lag_2 + 1)):
        time_window = matrix_STR.iloc[:, i:i+period_2].values.astype(np.float64)
        matrix_STR_lag.iloc[:, i + period_2 + lag_2 - 1] = np.nanmean(time_window, axis=1)
    Result = Align(matrix_STR_lag, BASE)
    return Result


def SEASON(BASE, EOD):
    """
    Category: Momentum/Seasonality/
    Factor Number: 12
    Factor Name: SEASON: Annual Seasonality
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN['ln(1+Rtn)'] = np.log(RTN['Rtn'] + 1)
    matrix_lnRTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='ln(1+Rtn)', columns='TRADE_DT')
    # Fill missing returns (due to no trading) with 0 for subsequent calculations
    matrix_lnRTN.fillna(0, inplace=True)
    matrix_Seasonality = pd.DataFrame(0, index=matrix_lnRTN.index, columns=matrix_lnRTN.columns)
    Y = 5
    for date in tqdm(matrix_Seasonality.columns):
        year = int(date[0:4])
        month = date[4:6]
        # Determine historical data corresponding to the same month for the past Y years
        history_dates = [str(year - i) + month for i in range(1, Y + 1)]
        efficiency = pd.Series(data=0, index=matrix_lnRTN.index)
        for history_date in history_dates:
            history_data = matrix_lnRTN.filter(regex='^' + history_date, axis=1)
            if not history_data.empty:
                monthly = np.cumprod(1 + history_data, axis=1).values[:, -1] - 1
                matrix_Seasonality.loc[:, date] += monthly
                efficiency += (monthly != 0)
        efficiency = efficiency.replace(0, 1)
        matrix_Seasonality.loc[:, date] = matrix_Seasonality.loc[:, date] / efficiency
    matrix_Seasonality = matrix_Seasonality.replace(0, np.nan)
    Result = Align(matrix_Seasonality, BASE)
    return Result


def INDMOM(I, EOD, EOD_DI):
    """
    Category: Momentum/Industry Momentum/
    Factor Number: 13
    Factor Name: INDMOM: Industry Momentum
    Inputs:
        I: Industry time series data for stocks.
        EOD: Data from the AShareEODPrices table.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    # For each stock, compute the weighted return and subtract the corresponding industry weighted return.
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN['ln(1+Rtn)'] = np.log(RTN['Rtn'] + 1)
    matrix_lnRTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='ln(1+Rtn)', columns='TRADE_DT')
    matrix_RSS = pd.DataFrame(index=matrix_lnRTN.index, columns=matrix_lnRTN.columns)
    m = len(matrix_RSS.index)
    n = len(matrix_RSS.columns)
    period = 21 * 6  # Window length (6 months)
    halflife = 21 * 1  # Half-life (1 month)
    weightSeries = getExponentialDecayWeight(halflife, period)
    weightSeries = np.matlib.repmat(weightSeries, m, 1)
    stocks = matrix_RSS.index
    # First, compute the weighted return series for individual stocks.
    for i in tqdm(range(n - period + 1)):
        matrix_RSS.iloc[:, i + period - 1] = np.nansum(matrix_lnRTN.iloc[:, i:i+period] * weightSeries, axis=1)
    matrix_RSS = matrix_RSS.replace(0, np.nan)
    matrix_INDMOM = pd.DataFrame(index=matrix_lnRTN.index, columns=matrix_lnRTN.columns)
    groupi = I.groupby(I["TRADE_DT"])
    groupmv = EOD_DI[["S_INFO_WINDCODE", "TRADE_DT", "S_DQ_MV"]].groupby(EOD_DI["TRADE_DT"])
    timelist = gettimelist(I)
    for date in tqdm(timelist):
        # For each trading day, get the industry and market cap series.
        industry = groupi.get_group(date)
        mv = groupmv.get_group(date)
        industry.index = industry['S_INFO_WINDCODE']
        mv.index = mv['S_INFO_WINDCODE']
        industry = pd.DataFrame(data=industry['INDUSTRIESNAME'], index=stocks)
        mv = pd.DataFrame(data=mv['S_DQ_MV'], index=stocks)
        mv['S_DQ_MV'] = mv['S_DQ_MV'].map(np.sqrt)
        # Convert industry labels to dummy variables.
        industry = pd.get_dummies(industry['INDUSTRIESNAME'])
        for col in industry.columns:
            # For each industry, compute the weighted average return using market cap weights.
            temp = industry[col] * mv['S_DQ_MV']
            s = np.nansum(temp * matrix_RSS[date]) / np.nansum(temp)
            industry[col] = industry[col] * s
        rsi = industry.sum(axis=1)
        matrix_INDMOM[date] = rsi - matrix_RSS[date]
    # (2) Apply a lagged window average
    matrix_INDMOM_lag = pd.DataFrame(index=matrix_INDMOM.index, columns=matrix_INDMOM.columns)
    period2 = 3  # Window length
    lag2 = 3     # Lag period
    for i in tqdm(range(n - period2 - lag2 + 1)):
        time_window = matrix_INDMOM.iloc[:, i:i+period2].values.astype(np.float64)
        matrix_INDMOM_lag.iloc[:, i + period2 + lag2 - 1] = np.nanmean(time_window, axis=1)
    BASE_temp = I[['S_INFO_WINDCODE', 'TRADE_DT']]  # Use I for alignment as it is the same as BASE
    Result = Align(matrix_INDMOM_lag, BASE_temp)
    return Result


def RSTR(BASE, EOD):
    """
    Category: Momentum/Momentum/
    Factor Number: 14
    Factor Name: RSTR: Relative Strength 12-month
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN['ln(1+Rtn)'] = np.log(RTN['Rtn'] + 1)
    RTN.reset_index(drop=True, inplace=True)
    # Convert individual stock return series into a 2D matrix for computation
    matrix_lnRTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='ln(1+Rtn)', columns='TRADE_DT')
    matrix_RSTR = pd.DataFrame(index=matrix_lnRTN.index, columns=matrix_lnRTN.columns)
    m = len(matrix_RSTR.index)
    n = len(matrix_RSTR.columns)
    period = 252  # Number of trading days
    halflife = 126
    weightSeries = getExponentialDecayWeight(halflife, period)
    weightSeries = np.matlib.repmat(weightSeries, m, 1)
    # (1) Compute non-lagged RSTR using exponential decay weights
    for i in tqdm(range(n - period + 1)):
        matrix_RSTR.iloc[:, i + period - 1] = np.nansum(matrix_lnRTN.iloc[:, i:i+period] * weightSeries, axis=1)
    matrix_RSTR = matrix_RSTR.replace(0, np.nan)
    # (2) Compute lagged window average (and take the opposite sign)
    matrix_RSTR_lag = pd.DataFrame(index=matrix_RSTR.index, columns=matrix_RSTR.columns)
    period = 11
    lag = 11
    for i in tqdm(range(n - period - lag + 1)):
        time_window = matrix_RSTR.iloc[:, i:i+period].values.astype(np.float64)
        matrix_RSTR_lag.iloc[:, i + period + lag - 1] = np.nanmean(time_window, axis=1)
    Result = Align(matrix_RSTR_lag, BASE)
    return Result


def HALPHA(BASE, EOD, INDEX_EOD):
    """
    Category: Momentum/Momentum/
    Factor Number: 15
    Factor Name: HALPHA: Historical Alpha
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
        INDEX_EOD: Data from the AindexEODPrices table.
    Output:
        Factor values.
    """
    # Get individual stock returns
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN.reset_index(drop=True, inplace=True)
    matrix_RTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    # Get benchmark index returns (China Securities 500 Index)
    MKT = INDEX_EOD[INDEX_EOD['S_INFO_WINDCODE'] == '000985.SH']
    MKT['Rtn'] = MKT['S_DQ_CLOSE'] / MKT['S_DQ_PRECLOSE'] - 1
    MKT.reset_index(drop=True, inplace=True)
    matrix_MKT = pd.pivot(MKT, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    matrix_ALPHA = pd.DataFrame(index=matrix_RTN.index, columns=matrix_RTN.columns)
    m = len(matrix_ALPHA.index)
    n = len(matrix_ALPHA.columns)
    period = 504
    halflife = 252
    weightSeries = getExponentialDecayWeight(halflife, period)
    # For each stock, regress its return series on the benchmark returns using exponential decay weights; take the constant term as the alpha
    for j in tqdm(range(n - period + 1)):
        X = matrix_MKT.iloc[:, j:j+period].T
        X = sm.add_constant(X)
        y = matrix_RTN.iloc[:, j:j+period]
        B = sm.WLS(y.T, X, weights=weightSeries.T).fit()
        matrix_ALPHA.iloc[:, j + period - 1] = B.params.values[0]
    # (2) Apply lagged window averaging and reverse sign
    matrix_ALPHA_lag = pd.DataFrame(index=matrix_ALPHA.index, columns=matrix_ALPHA.columns)
    period = 11
    lag = 11
    for i in tqdm(range(n - period - lag + 1)):
        time_window = matrix_ALPHA.iloc[:, i:i+period].values.astype(np.float64)
        matrix_ALPHA_lag.iloc[:, i + period + lag - 1] = np.nanmean(time_window, axis=1)
    Result = Align(matrix_ALPHA_lag, BASE)
    return Result


def MLEV(BASE, I, EOD_DI, BS):
    """
    Category: Quality/Leverage/
    Factor Number: 16
    Factor Name: MLEV: Market Leverage
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        I: Industry data for stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    # Filter annual report data (only year-end reports ending with '1231')
    BS = BS.loc[[True if i.endswith('1231') else False for i in BS['REPORT_PERIOD']]]
    # Get the intersection of stock codes between the original datasets
    stocks = sorted(set(BS["S_INFO_WINDCODE"]) & set(EOD_DI["S_INFO_WINDCODE"]))
    data_dict = {}
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    EOD_DI_group = EOD_DI.groupby(EOD_DI["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        BS_stock = BS_group.get_group(stock)
        EOD_DI_stock = EOD_DI_group.get_group(stock)
        # Select required fields and use ANN_DT_ALTER as the index
        tempBS = BS_stock[['ANN_DT_ALTER', 'REPORT_PERIOD', 'OTHER_EQUITY_TOOLS_P_SHR', 'TOT_NON_CUR_LIAB']]
        tempBS.index = tempBS['ANN_DT_ALTER']
        tempEOD_DI = EOD_DI_stock[['TRADE_DT', 'S_VAL_MV']]
        tempEOD_DI['S_VAL_MV'] = tempEOD_DI['S_VAL_MV'] * 10000  # S_VAL_MV is in 10,000 Yuan
        tempEOD_DI.index = tempEOD_DI['TRADE_DT']
        # Merge the data from different tables
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempBS, tempEOD_DI])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp.fillna(0, inplace=True)  # Fill missing values (e.g., for preferred shares) with 0
        # Calculate the factor value
        temp['MLEV'] = (temp['S_VAL_MV'] + temp['OTHER_EQUITY_TOOLS_P_SHR'] + temp['TOT_NON_CUR_LIAB']) / temp['S_VAL_MV']
        # For overlapping ANN_DT_ALTER, keep the most recent report update
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['MLEV']
    matrix = pd.DataFrame(data_dict).transpose()
    # Set factor values to NaN for banks, non-bank financials, and comprehensive financials
    df = matrix.stack().reset_index()
    df.columns = ['S_INFO_WINDCODE', 'TRADE_DT', 'MLEV']
    df = pd.merge(df, I, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])
    df.loc[df['INDUSTRIESNAME'] == '银行', 'MLEV'] = np.nan
    df.loc[df['INDUSTRIESNAME'] == '非银行金融', 'MLEV'] = np.nan
    df.loc[df['INDUSTRIESNAME'] == '综合金融 ', 'MLEV'] = np.nan
    matrix_MLEV = pd.pivot(df, index='S_INFO_WINDCODE', values='MLEV', columns='TRADE_DT')
    Result = Align(matrix_MLEV, BASE)
    return Result


def BLEV(BASE, I, BS):
    """
    Category: Quality/Leverage/
    Factor Number: 17
    Factor Name: BLEV: Book Leverage
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        I: Industry data for stocks.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    BS = BS.loc[[True if i.endswith('1231') else False for i in BS['REPORT_PERIOD']]]
    stocks = sorted(set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        BS_stock = BS_group.get_group(stock)
        temp = BS_stock[['ANN_DT_ALTER', 'REPORT_PERIOD', 'TOT_SHRHLDR_EQY_EXCL_MIN_INT', 'OTHER_EQUITY_TOOLS_P_SHR', 'TOT_NON_CUR_LIAB']]
        temp.index = temp['ANN_DT_ALTER']
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp.fillna(0, inplace=True)  # Fill missing values with 0
        temp['BLEV'] = (temp['TOT_SHRHLDR_EQY_EXCL_MIN_INT'] + temp['OTHER_EQUITY_TOOLS_P_SHR'] + temp['TOT_NON_CUR_LIAB']) / temp['TOT_SHRHLDR_EQY_EXCL_MIN_INT']
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['BLEV']
    matrix = pd.DataFrame(data_dict).transpose()
    df = matrix.stack().reset_index()
    df.columns = ['S_INFO_WINDCODE', 'TRADE_DT', 'BLEV']
    df = pd.merge(df, I, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])
    df.loc[df['INDUSTRIESNAME'] == '银行', 'BLEV'] = np.nan
    df.loc[df['INDUSTRIESNAME'] == '非银行金融', 'BLEV'] = np.nan
    df.loc[df['INDUSTRIESNAME'] == '综合金融 ', 'BLEV'] = np.nan
    matrix_BLEV = pd.pivot(df, index='S_INFO_WINDCODE', values='BLEV', columns='TRADE_DT')
    Result = Align(matrix_BLEV, BASE)
    return Result


def DTOA(BASE, I, BS):
    """
    Category: Quality/Leverage/
    Factor Number: 18
    Factor Name: DTOA: Debt-to-Assets
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        I: Industry data for stocks.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    BS = BS.loc[[True if i.endswith('1231') else False for i in BS['REPORT_PERIOD']]]
    stocks = sorted(set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        BS_stock = BS_group.get_group(stock)
        temp = BS_stock[['ANN_DT_ALTER', 'REPORT_PERIOD', 'TOT_ASSETS', 'TOT_LIAB']]
        temp.index = temp['ANN_DT_ALTER']
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp.fillna(0, inplace=True)
        temp['DTOA'] = temp['TOT_LIAB'] / temp['TOT_ASSETS']
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['DTOA']
    matrix = pd.DataFrame(data_dict).transpose()
    df = matrix.stack().reset_index()
    df.columns = ['S_INFO_WINDCODE', 'TRADE_DT', 'DTOA']
    df = pd.merge(df, I, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])
    df.loc[df['INDUSTRIESNAME'] == '银行', 'DTOA'] = np.nan
    df.loc[df['INDUSTRIESNAME'] == '非银行金融', 'DTOA'] = np.nan
    df.loc[df['INDUSTRIESNAME'] == '综合金融 ', 'DTOA'] = np.nan
    matrix_DTOA = pd.pivot(df, index='S_INFO_WINDCODE', values='DTOA', columns='TRADE_DT')
    Result = Align(matrix_DTOA, BASE)
    return Result


def VSAL(BASE, INCOME):
    """
    Category: Quality/Earnings Variability/
    Factor Number: 19
    Factor Name: VSAL: Variability in Sales
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        INCOME: Data from the AShareIncome table.
    Output:
        Factor values.
    """
    # Use only year-end report data
    INCOME = INCOME.loc[[True if i.endswith('1231') else False for i in INCOME['REPORT_PERIOD']]]
    stocks = sorted(set(INCOME["S_INFO_WINDCODE"]))
    data_dict = {}
    INCOME_group = INCOME.groupby(INCOME["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        INCOME_stock = INCOME_group.get_group(stock)
        tempINCOME = INCOME_stock[['OPER_REV', 'REPORT_PERIOD', 'ANN_DT_ALTER']]
        tempINCOME.index = tempINCOME['ANN_DT_ALTER']
        tempINCOME.sort_values(by=['REPORT_PERIOD'], inplace=True)
        temp = tempINCOME[['OPER_REV']]
        # Compute rolling 5-year standard deviation and mean, then calculate the ratio
        std = temp['OPER_REV'].rolling(5, min_periods=3).std()
        m = temp['OPER_REV'].rolling(5, min_periods=3).mean()
        temp['vis'] = std / m
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['vis']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def VERN(BASE, INCOME):
    """
    Category: Quality/Earnings Variability/
    Factor Number: 20
    Factor Name: VERN: Variability in Earnings
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        INCOME: Data from the AShareIncome table.
    Output:
        Factor values.
    """
    INCOME = INCOME.loc[[True if i.endswith('1231') else False for i in INCOME['REPORT_PERIOD']]]
    stocks = sorted(set(INCOME["S_INFO_WINDCODE"]))
    data_dict = {}
    INCOME_group = INCOME.groupby(INCOME["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        INCOME_stock = INCOME_group.get_group(stock)
        tempINCOME = INCOME_stock[['NET_PROFIT_AFTER_DED_NR_LP', 'REPORT_PERIOD', 'ANN_DT_ALTER']]
        tempINCOME.index = tempINCOME['ANN_DT_ALTER']
        tempINCOME.sort_values(by=['REPORT_PERIOD'], inplace=True)
        temp = tempINCOME[['NET_PROFIT_AFTER_DED_NR_LP']]
        std = temp['NET_PROFIT_AFTER_DED_NR_LP'].rolling(5, min_periods=3).std()
        m = temp['NET_PROFIT_AFTER_DED_NR_LP'].rolling(5, min_periods=3).mean()
        temp['vie'] = std / m
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['vie']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def VFLO(BASE, CF):
    """
    Category: Quality/Earnings Variability/
    Factor Number: 21
    Factor Name: VFLO: Variability in Cash-flows
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        CF: Data from the AShareCashFlow table.
    Output:
        Factor values.
    """
    CF = CF.loc[[True if i.endswith('1231') else False for i in CF['REPORT_PERIOD']]]
    stocks = sorted(set(CF["S_INFO_WINDCODE"]))
    data_dict = {}
    CF_group = CF.groupby(CF["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        CF_stock = CF_group.get_group(stock)
        tempCF = CF_stock.loc[:, ['NET_INCR_CASH_CASH_EQU', 'REPORT_PERIOD', 'ANN_DT_ALTER']]
        tempCF.index = tempCF['ANN_DT_ALTER']
        temp = tempCF[['NET_INCR_CASH_CASH_EQU']]
        std = temp['NET_INCR_CASH_CASH_EQU'].rolling(5, min_periods=3).std()
        m = temp['NET_INCR_CASH_CASH_EQU'].rolling(5, min_periods=3).mean()
        temp['vicf'] = std / m
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['vicf']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def ETOPF_STD(BASE, FEPS, EOD):
    """
    Category: Quality/Earnings Variability/
    Factor Number: 22
    Factor Name: ETOPF_STD: Standard deviation of Analyst Forecast Earnings-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FEPS: Data from the analyst eps_latest_report table.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    # Factor = std(predicted EPS) / price
    FEPS_std = FEPS.groupby(['S_INFO_WINDCODE', 'TRADE_DT'])['forecast_eps'].std().reset_index()
    data = pd.merge(FEPS_std, EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how='outer')
    data.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], inplace=True)
    data['STDFEP'] = data['forecast_eps'] / data['S_DQ_CLOSE']
    matrix_STDFEP = pd.pivot(data, index='S_INFO_WINDCODE', values='STDFEP', columns='TRADE_DT')
    Result = Align(matrix_STDFEP, BASE)
    return Result


def ABS(BASE, FI, CF, BS):
    """
    Category: Quality/Earnings Quality/
    Factor Number: 23
    Factor Name: ABS: Accruals - Balance Sheet Version
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FI: Data from the AShareFinancialIndicator table.
        CF: Data from the AShareCashFlow table.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    # Since depreciation and amortization data are only available in half-yearly and annual reports,
    # to ensure proper calculation, only use half-yearly and annual reports.
    FI = FI.loc[[True if i.endswith('0630') or i.endswith('1231') else False for i in FI['REPORT_PERIOD']]]
    CF = CF.loc[[True if i.endswith('0630') or i.endswith('1231') else False for i in CF['REPORT_PERIOD']]]
    BS = BS.loc[[True if i.endswith('0630') or i.endswith('1231') else False for i in BS['REPORT_PERIOD']]]
    # For cases with multiple records for the same ANN_DT_ALTER, take the latest record (reports are pre-sorted by REPORT_PERIOD)
    FI = FI.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    CF = CF.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    BS = BS.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    stocks = sorted(set(FI["S_INFO_WINDCODE"]) & set(CF["S_INFO_WINDCODE"]) & set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    FI_group = FI.groupby(FI["S_INFO_WINDCODE"])
    CF_group = CF.groupby(CF["S_INFO_WINDCODE"])
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        FI_stock = FI_group.get_group(stock)
        CF_stock = CF_group.get_group(stock)
        BS_stock = BS_group.get_group(stock)
        tempFI = FI_stock[['S_STM_IS', 'S_FA_EXINTERESTDEBT_CURRENT', 'S_FA_EXINTERESTDEBT_NONCURRENT']]
        tempFI.index = FI_stock['ANN_DT_ALTER']
        tempCF = CF_stock[['CASH_CASH_EQU_END_PERIOD']]
        tempCF.index = CF_stock['ANN_DT_ALTER']
        tempBS = BS_stock[['TOT_ASSETS', 'TOT_LIAB']]
        tempBS.index = BS_stock['ANN_DT_ALTER']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempFI, tempCF, tempBS])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp[['S_STM_IS', 'S_FA_EXINTERESTDEBT_CURRENT', 'S_FA_EXINTERESTDEBT_NONCURRENT', 'CASH_CASH_EQU_END_PERIOD']] = temp[['S_STM_IS', 'S_FA_EXINTERESTDEBT_CURRENT', 'S_FA_EXINTERESTDEBT_NONCURRENT', 'CASH_CASH_EQU_END_PERIOD']].fillna(0)
        noa = (temp['TOT_ASSETS'] - temp['CASH_CASH_EQU_END_PERIOD']) - (temp['TOT_LIAB'] - (temp['TOT_LIAB'] - temp['S_FA_EXINTERESTDEBT_CURRENT'] - temp['S_FA_EXINTERESTDEBT_NONCURRENT']))
        ACCR_BS = noa - noa.shift(1) - temp['S_STM_IS']
        data_dict[stock] = -ACCR_BS / temp['TOT_ASSETS']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def ACF(BASE, FI, CF, BS):
    """
    Category: Quality/Earnings Quality/
    Factor Number: 24
    Factor Name: ACF: Accruals - Cashflow Statement Version
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FI: Data from the AShareFinancialIndicator table.
        CF: Data from the AShareCashFlow table.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    FI = FI.loc[[True if i.endswith('0630') or i.endswith('1231') else False for i in FI['REPORT_PERIOD']]]
    CF = CF.loc[[True if i.endswith('0630') or i.endswith('1231') else False for i in CF['REPORT_PERIOD']]]
    BS = BS.loc[[True if i.endswith('0630') or i.endswith('1231') else False for i in BS['REPORT_PERIOD']]]
    FI = FI.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    CF = CF.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    BS = BS.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    stocks = sorted(set(FI["S_INFO_WINDCODE"]) & set(CF["S_INFO_WINDCODE"]) & set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    FI_group = FI.groupby(FI["S_INFO_WINDCODE"])
    CF_group = CF.groupby(CF["S_INFO_WINDCODE"])
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        FI_stock = FI_group.get_group(stock)
        CF_stock = CF_group.get_group(stock)
        BS_stock = BS_group.get_group(stock)
        tempFI = FI_stock[['S_STM_IS']]
        tempFI.index = FI_stock['ANN_DT_ALTER']
        tempCF = CF_stock[['NET_PROFIT', 'NET_CASH_FLOWS_OPER_ACT', 'NET_CASH_FLOWS_INV_ACT']]
        tempCF.index = CF_stock['ANN_DT_ALTER']
        tempBS = BS_stock[['TOT_ASSETS']]
        tempBS.index = BS_stock['ANN_DT_ALTER']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempFI, tempCF, tempBS])
        temp = temp.sort_index()
        temp[['S_STM_IS', 'NET_PROFIT', 'NET_CASH_FLOWS_OPER_ACT', 'NET_CASH_FLOWS_INV_ACT']] = temp[['S_STM_IS', 'NET_PROFIT', 'NET_CASH_FLOWS_OPER_ACT', 'NET_CASH_FLOWS_INV_ACT']].fillna(0)
        temp[['TOT_ASSETS']] = temp[['TOT_ASSETS']].fillna(method='ffill')
        accr = temp['NET_PROFIT'] - temp['NET_CASH_FLOWS_OPER_ACT'] - temp['NET_CASH_FLOWS_INV_ACT'] + temp['S_STM_IS']
        data_dict[stock] = -accr / temp['TOT_ASSETS']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def ATO(BASE, TTM, BS):
    """
    Category: Quality/Profitability/
    Factor Number: 25
    Factor Name: ATO: Asset Turnover
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        TTM: Data from the AShareTTMHis table.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    # Asset turnover = TTM operating revenue / Total assets
    TTM = TTM.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    BS = BS.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    stocks = sorted(set(TTM["S_INFO_WINDCODE"]) & set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    TTM_group = TTM.groupby(TTM["S_INFO_WINDCODE"])
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        TTM_stock = TTM_group.get_group(stock)
        BS_stock = BS_group.get_group(stock)
        tempTTM = TTM_stock[['OPER_REV_TTM']]
        tempTTM.index = TTM_stock['ANN_DT_ALTER']
        tempBS = BS_stock[['TOT_ASSETS']]
        tempBS.index = BS_stock['ANN_DT_ALTER']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempTTM, tempBS])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['AT'] = temp['OPER_REV_TTM'] / temp['TOT_ASSETS']
        data_dict[stock] = temp['AT']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def GP(BASE, INCOME, BS):
    """
    Category: Quality/Profitability/
    Factor Number: 26
    Factor Name: GP: Gross Profitability
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        INCOME: Data from the AShareIncome table.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    # Gross profitability = (Operating Revenue - Operating Cost) / Total Assets of the previous fiscal year
    INCOME = INCOME.loc[[True if i.endswith('1231') else False for i in INCOME['REPORT_PERIOD']]]
    BS = BS.loc[[True if i.endswith('1231') else False for i in BS['REPORT_PERIOD']]]
    stocks = sorted(set(INCOME["S_INFO_WINDCODE"]) & set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    INCOME_group = INCOME.groupby(INCOME["S_INFO_WINDCODE"])
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        INCOME_stock = INCOME_group.get_group(stock)
        BS_stock = BS_group.get_group(stock)
        tempINCOME = INCOME_stock[['ANN_DT_ALTER', 'REPORT_PERIOD', 'TOT_OPER_REV', 'TOT_OPER_COST']]
        tempINCOME.index = tempINCOME['ANN_DT_ALTER']
        tempBS = BS_stock[['ANN_DT_ALTER', 'REPORT_PERIOD', 'TOT_ASSETS']]
        tempBS.index = tempBS['ANN_DT_ALTER']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempINCOME, tempBS])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['GP'] = (temp['TOT_OPER_REV'] - temp['TOT_OPER_COST']) / temp['TOT_ASSETS']
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['GP']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def GPM(BASE, INCOME):
    """
    Category: Quality/Profitability/
    Factor Number: 27
    Factor Name: GPM: Gross Profit Margin
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        INCOME: Data from the AShareIncome table.
    Output:
        Factor values.
    """
    # Gross profit margin = (Operating Revenue - Operating Cost) / Operating Revenue of the previous fiscal year
    INCOME = INCOME.loc[[True if i.endswith('1231') else False for i in INCOME['REPORT_PERIOD']]]
    stocks = sorted(set(INCOME["S_INFO_WINDCODE"]))
    data_dict = {}
    INCOME_group = INCOME.groupby(INCOME["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        INCOME_stock = INCOME_group.get_group(stock)
        tempINCOME = INCOME_stock[['ANN_DT_ALTER', 'REPORT_PERIOD', 'TOT_OPER_REV', 'TOT_OPER_COST']]
        tempINCOME.index = tempINCOME['ANN_DT_ALTER']
        temp = tempINCOME.copy()
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['GPM'] = (temp['TOT_OPER_REV'] - temp['TOT_OPER_COST']) / temp['TOT_OPER_REV']
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['GPM']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def ROA(BASE, TTM, BS):
    """
    Category: Quality/Profitability/
    Factor Number: 28
    Factor Name: ROA: Return on Assets
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        TTM: Data from the AShareTTMHis table.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    # Return on Assets = TTM net profit / Total assets
    TTM = TTM.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    BS = BS.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    stocks = sorted(set(TTM["S_INFO_WINDCODE"]) & set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    TTM_group = TTM.groupby(TTM["S_INFO_WINDCODE"])
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        TTM_stock = TTM_group.get_group(stock)
        BS_stock = BS_group.get_group(stock)
        tempTTM = TTM_stock[['NET_PROFIT_TTM']]
        tempTTM.index = TTM_stock['ANN_DT_ALTER']
        tempBS = BS_stock[['TOT_ASSETS']]
        tempBS.index = BS_stock['ANN_DT_ALTER']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempTTM, tempBS])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['RoA'] = temp['NET_PROFIT_TTM'] / temp['TOT_ASSETS']
        data_dict[stock] = temp['RoA']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def AGRO(BASE, BS):
    """
    Category: Quality/Investment Quality/
    Factor Number: 29
    Factor Name: AGRO: Total Assets Growth Rate
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        BS: Data from the AShareBalanceSheet table.
    Output:
        Factor values.
    """
    # Use only annual report data
    BS = BS.loc[[True if i.endswith('1231') else False for i in BS['REPORT_PERIOD']]]
    stocks = sorted(set(BS["S_INFO_WINDCODE"]))
    data_dict = {}
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        BS_stock = BS_group.get_group(stock)
        tempBS = BS_stock.loc[:, ['TOT_ASSETS', 'REPORT_PERIOD', 'ANN_DT_ALTER']]
        tempBS.index = tempBS['ANN_DT_ALTER']
        tempBS.sort_values(by=['REPORT_PERIOD'], inplace=True)
        tempBS.fillna(method='ffill', inplace=True)
        temp = tempBS[['TOT_ASSETS']]
        temp['TIME'] = list(range(1, temp.shape[0] + 1))
        # Use rolling covariance (change corr to cov) to compute regression slope
        numerator = temp.rolling(5, min_periods=2).cov()
        numerator = numerator.xs('TOT_ASSETS', level=1)['TIME']
        slope = numerator / temp['TIME'].rolling(5, min_periods=2).var()
        temp['TAGR'] = -slope / (temp['TOT_ASSETS'].rolling(5, min_periods=2).mean())
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['TAGR']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def IGRO(BASE, CAP, TC, CL):
    """
    Category: Quality/Investment Quality/
    Factor Number: 30
    Factor Name: IGRO: Issuance Growth
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        CAP: Data from the AShareCapitalization table.
        TC: Data from the AShareTypeCode table.
        CL: Data from the AShareCalender table.
    Output:
        Factor values.
    """
    # Merge the capitalization table with the type code table to obtain the Chinese description of capital change type.
    # Here we only consider new issuance and repurchase.
    CAP = pd.merge(CAP, TC, left_on='S_SHARE_CHANGEREASON', right_on='S_TYPCODE', how='left')
    # Retain the last trading day of each year from BASE for annual capital changes
    CL['year'] = CL['TRADE_DAYS'].apply(lambda x: x[:4])
    CL = CL.groupby('year').last().reset_index()
    CL.rename(columns={'TRADE_DAYS': 'TRADE_DT'}, inplace=True)
    timelist = gettimelist(BASE)
    timelist_df = pd.DataFrame(timelist, columns=['TRADE_DT'])
    timelist_df = pd.merge(timelist_df, CL, on=['TRADE_DT'], how='inner')
    timelist_df.index = timelist_df['TRADE_DT']
    timelist_df.drop(['TRADE_DT', 'year'], axis=1, inplace=True)
    stocks = sorted(set(CAP["S_INFO_WINDCODE"]))
    data_dict = {}
    CAP_group = CAP.groupby(CAP["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        CAP_stock = CAP_group.get_group(stock)
        # Data may be inaccurate. For new issuance, sometimes the record is marked as "Other"
        # and for listed new issuance the record is marked as "New Issuance"; thus, the capital change
        # is identified when the total shares change.
        SHR_list = CAP_stock['TOT_SHR'].unique()
        CAP_stock['TOT_SHR_last'] = [SHR_list[np.where(SHR_list == i)[0][0] - 1] if np.where(SHR_list == i)[0][0] > 0 else np.nan for i in CAP_stock['TOT_SHR'].values]
        CAP_stock['ADJ_FACTOR'] = CAP_stock['TOT_SHR'] / CAP_stock['TOT_SHR_last']
        CAP_stock = CAP_stock[CAP_stock['S_TYPNAME'].isin(['增发', '回购'])]
        # Compute annual capital change
        tempCAP = CAP_stock[['ADJ_FACTOR']]
        tempCAP.index = CAP_stock['ANN_DT_ALTER']
        tempCAP = pd.merge(tempCAP, timelist_df, left_index=True, right_index=True, how='outer')
        tempCAP['year'] = [i[:4] for i in tempCAP.index]
        tempCAP['ADJ_FACTOR'].fillna(1, inplace=True)
        tempCAP['ADJ_FACTOR'] = tempCAP.groupby('year')['ADJ_FACTOR'].cumprod()
        tempCAP.drop_duplicates(subset='year', keep='last', inplace=True)
        tempCAP['TOT_SHR_ADJ'] = np.cumprod(tempCAP['ADJ_FACTOR'])
        temp = tempCAP[['TOT_SHR_ADJ']]
        temp['TIME'] = list(range(1, temp.shape[0] + 1))
        # Compute the regression slope with respect to time
        numerator = temp.rolling(5, min_periods=2).cov()
        numerator = numerator.xs('TOT_SHR_ADJ', level=1)['TIME']
        slope = numerator / temp['TIME'].rolling(5, min_periods=2).var()
        temp['IG'] = -slope / temp['TOT_SHR_ADJ'].rolling(5, min_periods=2).mean()
        data_dict[stock] = temp['IG']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def IGRO2(BASE, CAP):
    """
    An alternative algorithm for IGRO (considering all capital changes) for comparison.
    """
    timelist = gettimelist(BASE)
    timelist_df = pd.DataFrame(timelist, columns=['TRADE_DT'])
    timelist_df['year'] = timelist_df['TRADE_DT'].apply(lambda x: x[:4])
    timelist_df = timelist_df.groupby('year').last().reset_index()
    timelist_df.index = timelist_df['TRADE_DT']
    timelist_df.drop(['TRADE_DT', 'year'], axis=1, inplace=True)
    stocks = sorted(set(CAP["S_INFO_WINDCODE"]))
    data_dict = {}
    CAP_group = CAP.groupby(CAP["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        CAP_stock = CAP_group.get_group(stock)
        tempCAP = CAP_stock[['TOT_SHR']]
        tempCAP.index = CAP_stock['ANN_DT_ALTER']
        tempCAP = pd.merge(tempCAP, timelist_df, left_index=True, right_index=True, how='outer')
        tempCAP['year'] = [i[:4] for i in tempCAP.index]
        tempCAP['TOT_SHR'].fillna(method='ffill', inplace=True)
        tempCAP.drop_duplicates(subset='year', keep='last', inplace=True)
        temp = tempCAP[['TOT_SHR']]
        temp['TIME'] = list(range(1, temp.shape[0] + 1))
        numerator = temp.rolling(5, min_periods=2).cov()
        numerator = numerator.xs('TOT_SHR', level=1)['TIME']
        slope = numerator / temp['TIME'].rolling(5, min_periods=2).var()
        temp['IG'] = -slope / temp['TOT_SHR'].rolling(5, min_periods=2).mean()
        data_dict[stock] = temp['IG']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def CXGRO(BASE, CF):
    """
    Category: Quality/Investment Quality/
    Factor Number: 31
    Factor Name: CXGRO: Capital Expenditure Growth
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        CF: Data from the AShareCashFlow table.
    Output:
        Factor values.
    """
    CF = CF.loc[[True if i.endswith('1231') else False for i in CF['REPORT_PERIOD']]]
    stocks = sorted(set(CF["S_INFO_WINDCODE"]))
    data_dict = {}
    CF_group = CF.groupby(CF["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        CF_stock = CF_group.get_group(stock)
        tempCF = CF_stock.loc[:, ['ANN_DT_ALTER', 'REPORT_PERIOD', 'CASH_PAY_ACQ_CONST_FIOLTA']]
        tempCF.index = tempCF['ANN_DT_ALTER']
        tempCF.sort_values(by=['REPORT_PERIOD'], inplace=True)
        tempCF.fillna(method='ffill', inplace=True)
        temp = tempCF[['CASH_PAY_ACQ_CONST_FIOLTA']]
        temp['TIME'] = list(range(1, temp.shape[0] + 1))
        numerator = temp.rolling(5, min_periods=2).cov()
        numerator = numerator.xs('CASH_PAY_ACQ_CONST_FIOLTA', level=1)['TIME']
        slope = numerator / temp['TIME'].rolling(5, min_periods=2).var()
        temp['CEG'] = -slope / (temp['CASH_PAY_ACQ_CONST_FIOLTA'].rolling(5, min_periods=2).mean())
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['CEG']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def BTOP(BASE, EOD_DI):
    """
    Category: Value/Book-to-Price/
    Factor Number: 32
    Factor Name: BTOP: Book-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    # Use the reciprocal of S_VAL_PB_NEW from the EOD_DI table
    BP = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_VAL_PB_NEW']]
    BP['BP'] = 1 / BP['S_VAL_PB_NEW']
    BP = BP[['S_INFO_WINDCODE', 'TRADE_DT', 'BP']]
    BP.reset_index(drop=True, inplace=True)  # Reset index before pivot_table for pandas 2.0
    matrix_BP = pd.pivot(BP, index='S_INFO_WINDCODE', values='BP', columns='TRADE_DT')
    Result = Align(matrix_BP, BASE)
    return Result


def ETOP(BASE, TTM, EOD_DI):
    """
    Category: Value/Earnings Yield/
    Factor Number: 33
    Factor Name: ETOP: Earnings-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        TTM: Data from the AShareTTMHis table.
        EOD_DI: Data from the EODDerivativeIndicator table.
    Output:
        Factor values.
    """
    # Earnings-to-Price: TTM earnings divided by current market capitalization
    TTM = TTM.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    stocks = sorted(set(TTM["S_INFO_WINDCODE"]) & set(EOD_DI["S_INFO_WINDCODE"]))
    data_dict = {}
    TTM_group = TTM.groupby(TTM["S_INFO_WINDCODE"])
    EOD_DI_group = EOD_DI.groupby(EOD_DI["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        TTM_stock = TTM_group.get_group(stock)
        EOD_DI_stock = EOD_DI_group.get_group(stock)
        tempTTM = TTM_stock[['NET_PROFIT_TTM']]  # NET_PROFIT_TTM is in Yuan
        tempTTM.index = TTM_stock['ANN_DT_ALTER']
        tempEOD_DI = EOD_DI_stock[['S_VAL_MV']] * 10000  # S_VAL_MV is in 10,000 Yuan
        tempEOD_DI.index = EOD_DI_stock['TRADE_DT']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempTTM, tempEOD_DI])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['EP'] = temp['NET_PROFIT_TTM'] / temp['S_VAL_MV']
        data_dict[stock] = temp['EP']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def ETOPF(BASE, FEP):
    """
    Category: Value/Earnings Yield/
    Factor Number: 34
    Factor Name: ETOPF: Analyst-Predicted Earnings-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FEP: Data from the analyst pe_latest_report table.
    Output:
        Factor values.
    """
    stocks = sorted(set(FEP["S_INFO_WINDCODE"]))
    data_dict = {}
    FEP_group = FEP.groupby(FEP["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        FEP_stock = FEP_group.get_group(stock)
        # Compute the average of multiple predicted PE values
        tempFEP = FEP_stock.groupby(['S_INFO_WINDCODE', 'TRADE_DT'])['forecast_ep'].mean().reset_index()
        tempFEP.index = tempFEP['TRADE_DT']
        temp = tempFEP[['forecast_ep']]
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        data_dict[stock] = temp['forecast_ep']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def CETOP(BASE, TTM, EOD_DI):
    """
    Category: Value/Earnings Yield/
    Factor Number: 35
    Factor Name: CETOP: Cash-Earnings-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        TTM: Data from the AShareTTMHis table.
        EOD_DI: Data from the AShareEODDerivativeIndicator table.
    Output:
        Factor values.
    """
    # Cash-Earnings-to-Price: TTM cash earnings divided by current market cap
    TTM = TTM.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    stocks = sorted(set(TTM["S_INFO_WINDCODE"]) & set(EOD_DI["S_INFO_WINDCODE"]))
    data_dict = {}
    TTM_group = TTM.groupby(TTM["S_INFO_WINDCODE"])
    EOD_DI_group = EOD_DI.groupby(EOD_DI["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        TTM_stock = TTM_group.get_group(stock)
        EOD_DI_stock = EOD_DI_group.get_group(stock)
        tempTTM = TTM_stock[['NET_CASH_FLOWS_OPER_ACT_TTM']]
        tempTTM.index = TTM_stock['ANN_DT_ALTER']
        tempEOD_DI = EOD_DI_stock[['S_VAL_MV']] * 10000  # S_VAL_MV in 10,000 Yuan
        tempEOD_DI.index = EOD_DI_stock['TRADE_DT']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempTTM, tempEOD_DI])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['CEP'] = temp['NET_CASH_FLOWS_OPER_ACT_TTM'] / temp['S_VAL_MV']
        data_dict[stock] = temp['CEP']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def EM(BASE, INCOME, EOD_DI, BS, CF):
    """
    Category: Value/Earnings Yield/
    Factor Number: 36
    Factor Name: EM: Enterprise Multiple (EBIT to EV)
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        INCOME: Data from the AShareIncome table.
        EOD_DI: Data from the AShareEODDerivativeIndicator table.
        BS: Data from the AShareBalanceSheet table.
        CF: Data from the AShareCashFlow table.
    Output:
        Factor values.
    """
    # Enterprise Multiple = EBIT (from previous fiscal year) / Current Enterprise Value,
    # where EV = market cap + total liabilities - ending cash.
    INCOME = INCOME.loc[[True if i.endswith('1231') else False for i in INCOME['REPORT_PERIOD']]]
    BS = BS.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    CF = CF.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    stocks = sorted(set(INCOME["S_INFO_WINDCODE"]) & set(EOD_DI["S_INFO_WINDCODE"]) & set(BS["S_INFO_WINDCODE"]) & set(CF["S_INFO_WINDCODE"]))
    data_dict = {}
    INCOME_group = INCOME.groupby(INCOME["S_INFO_WINDCODE"])
    EODDI_group = EOD_DI.groupby(EOD_DI["S_INFO_WINDCODE"])
    BS_group = BS.groupby(BS["S_INFO_WINDCODE"])
    CF_group = CF.groupby(CF["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        INCOME_stock = INCOME_group.get_group(stock)
        EODDI_stock = EODDI_group.get_group(stock)
        BS_stock = BS_group.get_group(stock)
        CF_stock = CF_group.get_group(stock)
        tempINCOME = INCOME_stock[['EBIT']]
        tempINCOME.index = INCOME_stock['ANN_DT_ALTER']
        tempEODDI = EODDI_stock[['S_DQ_MV']]
        tempEODDI.index = EODDI_stock['TRADE_DT']
        tempBS = BS_stock[['TOT_LIAB']]
        tempBS.index = BS_stock['ANN_DT_ALTER']
        tempCF = CF_stock[['CASH_CASH_EQU_END_PERIOD']]
        tempCF.index = CF_stock['ANN_DT_ALTER']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempINCOME, tempEODDI, tempBS, tempCF])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['EM'] = temp['EBIT'] / (temp['S_DQ_MV'] + temp['TOT_LIAB'] - temp['CASH_CASH_EQU_END_PERIOD'])
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['EM']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def LTRSTR(BASE, EOD):
    """
    Category: Value/Long-Term Reversal/
    Factor Number: 37
    Factor Name: LTRSTR: Long-term Relative Strength
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    RTN['ln(1+Rtn)'] = np.log(RTN['Rtn'] + 1)
    matrix_lnRTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='ln(1+Rtn)', columns='TRADE_DT')
    matrix_NLLRS = pd.DataFrame(index=matrix_lnRTN.index, columns=matrix_lnRTN.columns)
    m = len(matrix_NLLRS.index)
    n = len(matrix_NLLRS.columns)
    # (1) Compute non-lagged long-term relative strength
    period_1 = 1040  # Window length
    halflife_1 = 260  # Half-life
    weightSeries_1 = getExponentialDecayWeight(halflife_1, period_1)
    weightSeries_1 = np.matlib.repmat(weightSeries_1, m, 1)
    for i in tqdm(range(n - period_1 + 1)):
        matrix_NLLRS.iloc[:, i + period_1 - 1] = np.nansum(matrix_lnRTN.iloc[:, i: i + period_1] * weightSeries_1, axis=1)
    matrix_NLLRS = matrix_NLLRS.replace(0, np.nan)
    # (2) Apply lagged window averaging and reverse sign
    matrix_LTRS = pd.DataFrame(index=matrix_lnRTN.index, columns=matrix_lnRTN.columns)
    period_2 = 11
    lag_2 = 273
    for i in tqdm(range(n - period_2 - lag_2 + 1)):
        time_window = matrix_NLLRS.iloc[:, i:i+period_2].values.astype(np.float64)
        matrix_LTRS.iloc[:, i + period_2 + lag_2 - 1] = -np.nanmean(time_window, axis=1)
    Result = Align(matrix_LTRS, BASE)
    return Result


def LTHALPHA(BASE, EOD, INDEX_EOD):
    """
    Category: Value/Long-Term Reversal/
    Factor Number: 38
    Factor Name: LTHALPHA: Long-term Historical Alpha
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        EOD: Data from the AShareEODPrices table.
        INDEX_EOD: Data from the AindexEODPrices table.
    Output:
        Factor values.
    """
    RTN = EOD[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE']]
    RTN['Rtn'] = RTN['S_DQ_CLOSE'] / RTN['S_DQ_PRECLOSE'] - 1
    matrix_RTN = pd.pivot(RTN, index='S_INFO_WINDCODE', values='Rtn', columns='TRADE_DT')
    # Get benchmark index returns
    INDEX_RTN = INDEX_EOD[INDEX_EOD['S_INFO_WINDCODE'] == '000985.SH']
    INDEX_RTN['RTN'] = INDEX_RTN['S_DQ_CLOSE'] / INDEX_RTN['S_DQ_PRECLOSE'] - 1
    INDEX_RTN.reset_index(drop=True, inplace=True)
    matrix_INDEX_RTN = pd.pivot(INDEX_RTN, index='S_INFO_WINDCODE', values='RTN', columns='TRADE_DT')
    # Align the start and end dates of matrix_RTN and matrix_INDEX_RTN
    start_date = max(matrix_RTN.columns[0], matrix_INDEX_RTN.columns[0])
    end_date = min(matrix_RTN.columns[-1], matrix_INDEX_RTN.columns[-1])
    matrix_RTN = matrix_RTN.loc[:, (matrix_RTN.columns >= start_date) & (matrix_RTN.columns <= end_date)]
    matrix_INDEX_RTN = matrix_INDEX_RTN.loc[:, (matrix_INDEX_RTN.columns >= start_date) & (matrix_INDEX_RTN.columns <= end_date)]
    matrix_ALPHA = pd.DataFrame(index=matrix_RTN.index, columns=matrix_RTN.columns)
    m = len(matrix_ALPHA.index)
    n = len(matrix_ALPHA.columns)
    period_1 = 1040
    halflife_1 = 260
    weightSeries = getExponentialDecayWeight(halflife_1, period_1)
    # For each stock, run a weighted regression of its return series on the benchmark returns; take the constant as alpha
    for i in tqdm(range(n - period_1 + 1)):
        X = matrix_INDEX_RTN.iloc[:, i:i+period_1].T
        X = sm.add_constant(X)
        y = matrix_RTN.iloc[:, i:i+period_1]
        Reg = sm.WLS(y.T, X, weights=weightSeries.T).fit()
        matrix_ALPHA.iloc[:, i + period_1 - 1] = Reg.params.values[0]
    # (2) Apply lagged window averaging and reverse sign
    matrix_LTHA = pd.DataFrame(index=matrix_ALPHA.index, columns=matrix_ALPHA.columns)
    period_2 = 11
    lag_2 = 273
    for i in tqdm(range(n - period_2 - lag_2 + 1)):
        time_window = matrix_ALPHA.iloc[:, i:i+period_2].values.astype(np.float64)
        matrix_LTHA.iloc[:, i + period_2 + lag_2 - 1] = -np.nanmean(time_window, axis=1)
    Result = Align(matrix_LTHA, BASE)
    return Result


def EGRLF(BASE, NP_FY1, NP_FY2, NP_FY3):
    """
    Category: Growth/Growth/
    Factor Number: 39
    Factor Name: EGRLF: Analyst Predicted Earnings Long-term Growth
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        NP_FY1: Data from the analyst np_fy1 table.
        NP_FY2: Data from the analyst np_fy2 table.
        NP_FY3: Data from the analyst np_fy3 table.
    Output:
        Factor values.
    """
    stocks = sorted(set(NP_FY1["S_INFO_WINDCODE"]) & set(NP_FY2["S_INFO_WINDCODE"]) & set(NP_FY3["S_INFO_WINDCODE"]))
    data_dict = {}
    NP_FY1_group = NP_FY1.groupby(NP_FY1["S_INFO_WINDCODE"])
    NP_FY2_group = NP_FY2.groupby(NP_FY2["S_INFO_WINDCODE"])
    NP_FY3_group = NP_FY3.groupby(NP_FY3["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        NP_FY1_stock = NP_FY1_group.get_group(stock)
        NP_FY2_stock = NP_FY2_group.get_group(stock)
        NP_FY3_stock = NP_FY3_group.get_group(stock)
        tempNP_FY1 = NP_FY1_stock.groupby(['S_INFO_WINDCODE', 'TRADE_DT'])['np_fy1'].mean().reset_index()
        tempNP_FY1.index = tempNP_FY1['TRADE_DT']
        tempNP_FY1 = tempNP_FY1[['np_fy1']]
        tempNP_FY2 = NP_FY2_stock.groupby(['S_INFO_WINDCODE', 'TRADE_DT'])['np_fy2'].mean().reset_index()
        tempNP_FY2.index = tempNP_FY2['TRADE_DT']
        tempNP_FY2 = tempNP_FY2[['np_fy2']]
        tempNP_FY3 = NP_FY3_stock.groupby(['S_INFO_WINDCODE', 'TRADE_DT'])['np_fy3'].mean().reset_index()
        tempNP_FY3.index = tempNP_FY3['TRADE_DT']
        tempNP_FY3 = tempNP_FY3[['np_fy3']]
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
                      [tempNP_FY1, tempNP_FY2, tempNP_FY3])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        beta = (temp['np_fy3'] - temp['np_fy1']) / 2
        beta = beta.replace(np.nan, 0)
        beta += (beta == 0) * (temp['np_fy2'] - temp['np_fy1'])
        growth = beta / temp.mean(axis=1, skipna=True)
        data_dict[stock] = growth
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def EGRO(BASE, INCOME):
    """
    Category: Growth/Growth/
    Factor Number: 40
    Factor Name: EGRO: Earnings per Share Growth Rate
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        INCOME: Data from the AShareIncome table.
    Output:
        Factor values.
    """
    INCOME = INCOME.loc[[True if i.endswith('1231') else False for i in INCOME['REPORT_PERIOD']]]
    stocks = sorted(set(INCOME["S_INFO_WINDCODE"]))
    data_dict = {}
    INCOME_group = INCOME.groupby(INCOME["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        INCOME_stock = INCOME_group.get_group(stock)
        tempINCOME = INCOME_stock[['NET_PROFIT_AFTER_DED_NR_LP', 'REPORT_PERIOD', 'ANN_DT_ALTER']]
        tempINCOME.index = tempINCOME['ANN_DT_ALTER']
        tempINCOME.sort_values(by=['REPORT_PERIOD'], inplace=True)
        tempINCOME.fillna(method='ffill', inplace=True)
        temp = tempINCOME[['NET_PROFIT_AFTER_DED_NR_LP']]
        temp['TIME'] = list(range(1, temp.shape[0] + 1))
        numerator = temp.rolling(5, min_periods=2).cov()
        numerator = numerator.xs('NET_PROFIT_AFTER_DED_NR_LP', level=1)['TIME']
        slope = numerator / temp['TIME'].rolling(5, min_periods=2).var()
        temp['EPSG'] = slope / (temp['NET_PROFIT_AFTER_DED_NR_LP'].rolling(5, min_periods=2).mean())
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['EPSG']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def SGRO_OLS(df, X):
    """
    Helper function for SGRO: performs OLS regression on a given dataframe.
    """
    Y = df['OPER_REV']
    for j in range(len(Y) - 16):
        y = df['OPER_REV'][j:j+20:4]
        # If more than 2 of the 5 values are missing, skip the regression.
        if np.nansum(np.isnan(y)) > 2:
            continue
        # Otherwise, the factor value is the regression slope over the past five years divided by the average absolute value.
        model = sm.OLS(y, X, missing='drop').fit()
        df.loc[max(y.index), 'SGRO_new'] = model.params.values[0] / np.mean(np.abs(y))
    return df


def SGRO(BASE, INCOME):
    """
    Category: Growth/Growth/
    Factor Number: 41
    Factor Name: SGRO: Sales per Share Growth Rate
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        INCOME: Data from the AShareIncome table.
    Output:
        Factor values.
    """
    INCOME = INCOME.loc[[True if i.endswith('1231') else False for i in INCOME['REPORT_PERIOD']]]
    stocks = sorted(set(INCOME["S_INFO_WINDCODE"]))
    data_dict = {}
    INCOME_group = INCOME.groupby(INCOME["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        INCOME_stock = INCOME_group.get_group(stock)
        tempINCOME = INCOME_stock[['OPER_REV', 'REPORT_PERIOD', 'ANN_DT_ALTER']]
        tempINCOME.index = tempINCOME['ANN_DT_ALTER']
        tempINCOME.sort_values(by=['REPORT_PERIOD'], inplace=True)
        tempINCOME.fillna(method='ffill', inplace=True)
        temp = tempINCOME[['OPER_REV']]
        temp['TIME'] = list(range(1, temp.shape[0] + 1))
        numerator = temp.rolling(5, min_periods=2).cov()
        numerator = numerator.xs('OPER_REV', level=1)['TIME']
        slope = numerator / temp['TIME'].rolling(5, min_periods=2).var()
        temp['SPSG'] = slope / (temp['OPER_REV'].rolling(5, min_periods=2).mean())
        temp_drop_duplicates = temp.loc[~temp.index.duplicated(keep='last')]
        data_dict[stock] = temp_drop_duplicates['SPSG']
    matrix = pd.DataFrame(data_dict).transpose()
    SGRO_1 = matrix.reset_index(drop=False)
    matrix_SGRO = pd.pivot_table(SGRO_1, index='S_INFO_WINDCODE', values='OPER_REV', columns='TRADE_DT', dropna=False)
    Result = Align(matrix_SGRO, BASE)
    return Result


def RR(BASE, FEA):
    """
    Category: Sentiment/Analyst Sentiment/
    Factor Number: 42
    Factor Name: RR: Revision Ratio
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FEA: Data from the analyst rpt_rating_adjust table.
    Output:
        Factor values.
    """
    stocks = sorted(set(FEA["S_INFO_WINDCODE"]))
    data_dict = {}
    timelist = gettimelist(BASE)
    # Create a list of unique months from the base data
    monthlist = sorted(set([i[:6] for i in timelist]))
    FEA_group = FEA.groupby('S_INFO_WINDCODE')
    for stock in tqdm(stocks):
        FEA_stock = FEA_group.get_group(stock)
        FEA_stock['month'] = FEA_stock['ANN_DT_ALTER'].apply(lambda x: x[:6])
        # (1) Calculate monthly rating adjustments
        def Cal_Ratio(group):
            Ratio = ((group['np_adjust_mark'] == 1).sum() - (group['np_adjust_mark'] == -1).sum()) / \
                    ((group['np_adjust_mark'] == 1).sum() + (group['np_adjust_mark'] == -1).sum())
            return Ratio
        temp = pd.DataFrame(FEA_stock.groupby('month').apply(Cal_Ratio), columns=['Ratio'])
        dfMonth = pd.DataFrame(index=monthlist)
        temp = pd.merge(temp, dfMonth, left_index=True, right_index=True, how='right')
        # The computed ratio for each month is used for the next month factor value
        temp.index = temp.index.map(lambda x: next_month(x))
        temp['RRIBS'] = np.nan
        # (2) Apply 3-month weighted smoothing
        period = 3
        half_life = 1
        weights = getExponentialDecayWeight(half_life, period)
        for i in range(period - 1, len(temp)):
            temp.loc[temp.index[i], 'RRIBS'] = np.sum(weights * temp.iloc[i - period + 1:i + 1]['Ratio'])
        dfDate = pd.DataFrame(timelist, columns=['TRADE_DT'])
        dfDate.index = dfDate['TRADE_DT'].apply(lambda x: x[:6])
        temp = pd.merge(temp, dfDate, left_index=True, right_index=True, how='right')
        temp.index = temp['TRADE_DT']
        data_dict[stock] = temp['RRIBS']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def ETOPF_C(BASE, FEP):
    """
    Category: Sentiment/Analyst Sentiment/
    Factor Number: 43
    Factor Name: ETOPF_C: Change in Analyst-Predicted Earnings-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FEP: Data from the analyst pe_latest_report table.
    Output:
        Factor values.
    """
    stocks = sorted(set(FEP["S_INFO_WINDCODE"]))
    data_dict = {}
    timelist = gettimelist(BASE)
    FEP_group = FEP.groupby('S_INFO_WINDCODE')
    for stock in tqdm(stocks):
        FEP_stock = FEP_group.get_group(stock)
        FEP_stock['month'] = FEP_stock['TRADE_DT'].apply(lambda x: x[:6])
        # (1) Calculate monthly change in predicted EP
        temp = pd.DataFrame(FEP_stock.groupby('month')['forecast_ep'].mean(), columns=['forecast_ep'])
        temp['forecast_ep_chg'] = temp['forecast_ep'] / temp['forecast_ep'].shift(1) - 1
        temp.index = temp.index.map(lambda x: next_month(x))
        temp['EPIBS'] = np.nan
        # (2) Apply 4-month weighted smoothing
        period = 4
        half_life = 1
        weights = getExponentialDecayWeight(half_life, period)
        for i in range(period - 1, len(temp)):
            temp.loc[temp.index[i], 'EPIBS'] = np.sum(weights * temp.iloc[i - period + 1:i + 1]['forecast_ep_chg'])
        dfDate = pd.DataFrame(timelist, columns=['TRADE_DT'])
        dfDate.index = dfDate['TRADE_DT'].apply(lambda x: x[:6])
        temp = pd.merge(temp, dfDate, left_index=True, right_index=True, how='right')
        temp.index = temp['TRADE_DT']
        temp = temp.replace(np.inf, np.nan)
        temp = temp.replace(-np.inf, np.nan)
        data_dict[stock] = temp['EPIBS']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def EPSF_C(BASE, FEPS):
    """
    Category: Sentiment/Analyst Sentiment/
    Factor Number: 44
    Factor Name: EPSF_C: Change in Analyst-Predicted Earnings per Share
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FEPS: Data from the analyst eps_latest_report table.
    Output:
        Factor values.
    """
    stocks = sorted(set(FEPS["S_INFO_WINDCODE"]))
    data_dict = {}
    timelist = gettimelist(BASE)
    FEPS_group = FEPS.groupby('S_INFO_WINDCODE')
    for stock in tqdm(stocks):
        FEPS_stock = FEPS_group.get_group(stock)
        FEPS_stock['month'] = FEPS_stock['TRADE_DT'].apply(lambda x: x[:6])
        # (1) Calculate monthly change in predicted EPS
        temp = pd.DataFrame(FEPS_stock.groupby('month')['forecast_eps'].mean(), columns=['forecast_eps'])
        temp['forecast_eps_chg'] = temp['forecast_eps'] / temp['forecast_eps'].shift(1) - 1
        temp.index = temp.index.map(lambda x: next_month(x))
        temp['EARN'] = np.nan
        # (2) Apply 4-month weighted smoothing
        period = 4
        half_life = 1
        weights = getExponentialDecayWeight(half_life, period)
        for i in range(period - 1, len(temp)):
            temp.loc[temp.index[i], 'EARN'] = np.sum(weights * temp.iloc[i - period + 1:i + 1]['forecast_eps_chg'])
        dfDate = pd.DataFrame(timelist, columns=['TRADE_DT'])
        dfDate.index = dfDate['TRADE_DT'].apply(lambda x: x[:6])
        temp = pd.merge(temp, dfDate, left_index=True, right_index=True, how='right')
        temp.index = temp['TRADE_DT']
        temp = temp.replace(np.inf, np.nan)
        temp = temp.replace(-np.inf, np.nan)
        data_dict[stock] = temp['EARN']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def DTOP(BASE, AllPeriod, EOD, DVD):
    """
    Category: Dividend Yield/Dividend Yield/
    Factor Number: 45
    Factor Name: DTOP: Dividend-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        AllPeriod: Announcement dates for all financial reports.
        EOD: Data from the AShareEODPrices table.
        DVD: Data from the AShareDividend table.
    Output:
        Factor values.
    """
    # In A-shares, dividends are relatively rare, so many stocks have 0 dividend.
    # Typically, dividend yield = last fiscal year's total dividend / month-end price.
    # Dividends are usually announced in half-yearly or annual reports.
    EOD['month'] = EOD['TRADE_DT'].apply(lambda x: x[:6])
    EOD_month = EOD.groupby(['S_INFO_WINDCODE', 'month']).last().reset_index()
    # Keep only records with status "Implemented"
    DVD = DVD[DVD["S_DIV_PROGRESS"] == '3']
    # Some records are stock dividends rather than cash dividends; retain only records with nonzero cash dividend per share.
    DVD = DVD[DVD["CASH_DVD_PER_SH_AFTER_TAX"] != 0]
    # Merge AllPeriod with DVD to unify fiscal year dividends
    AllPeriod = AllPeriod[[True if AllPeriod['REPORT_PERIOD'][i].endswith('1231') else False for i in range(len(AllPeriod))]]
    AllPeriod = AllPeriod.groupby(['S_INFO_WINDCODE', 'ANN_DT_ALTER']).last().reset_index()
    DVD = pd.merge(DVD, AllPeriod, on=['S_INFO_WINDCODE', 'ANN_DT_ALTER', 'REPORT_PERIOD'], how='outer').sort_values(by=['S_INFO_WINDCODE', 'ANN_DT_ALTER', 'REPORT_PERIOD'])
    DVD["CASH_DVD_PER_SH_AFTER_TAX"].fillna(0, inplace=True)
    stocks = sorted(set(EOD_month["S_INFO_WINDCODE"]) & set(DVD["S_INFO_WINDCODE"]))
    data_dict = {}
    EOD_group = EOD_month.groupby(EOD_month["S_INFO_WINDCODE"])
    DVD_group = DVD.groupby(DVD["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        EOD_stock = EOD_group.get_group(stock)
        DVD_stock = DVD_group.get_group(stock)
        EOD_stock['year'] = EOD_stock['TRADE_DT'].apply(lambda x: x[:4])
        DVD_stock['year'] = DVD_stock['REPORT_PERIOD'].apply(lambda x: x[:4])
        # Compute total dividend per fiscal year using the last ANN_DT_ALTER of that year
        tempDVD = pd.DataFrame(DVD_stock.groupby('year')['CASH_DVD_PER_SH_AFTER_TAX'].sum(), columns=['CASH_DVD_PER_SH_AFTER_TAX'])
        tempDVD.index = DVD_stock.groupby('year').last()['ANN_DT_ALTER']
        # Typically, dividend yield can only be calculated when the prior fiscal year's report is announced.
        tempDVD['year'] = DVD_stock.groupby('year').last().index
        tempDVD = tempDVD[[False if tempDVD.index[i][:4] == tempDVD['year'].values[i] else True for i in range(len(tempDVD))]]
        tempDVD = tempDVD[[False if str(int(tempDVD.index[i][:4]) - 2) == tempDVD['year'].values[i] else True for i in range(len(tempDVD))]]
        tempDVD.drop(columns=['year'], inplace=True)
        tempEOD = EOD_stock[['S_DQ_CLOSE']]
        tempEOD.index = EOD_stock['TRADE_DT']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempDVD, tempEOD])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['DTPR'] = temp['CASH_DVD_PER_SH_AFTER_TAX'] / temp['S_DQ_CLOSE']
        data_dict[stock] = temp['DTPR']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


def DTOPF(BASE, FDPS, EOD):
    """
    Category: Dividend Yield/Dividend Yield/
    Factor Number: 46
    Factor Name: DTOPF: Analyst-Predicted Dividend-to-Price
    Inputs:
        BASE: Base data containing all trading days and tradable stocks.
        FDPS: Data from the analyst dps_latest_report table.
        EOD: Data from the AShareEODPrices table.
    Output:
        Factor values.
    """
    # Use the analyst predicted DPS divided by current price directly.
    stocks = sorted(set(FDPS["S_INFO_WINDCODE"]) & set(EOD["S_INFO_WINDCODE"]))
    data_dict = {}
    FDPS_group = FDPS.groupby(FDPS["S_INFO_WINDCODE"])
    EOD_group = EOD.groupby(EOD["S_INFO_WINDCODE"])
    for stock in tqdm(stocks):
        FDPS_stock = FDPS_group.get_group(stock)
        EOD_stock = EOD_group.get_group(stock)
        # Preprocess FDPS to remove outliers (discard extreme values)
        min_mask = FDPS_stock["forecast_dps"] < (FDPS_stock["forecast_dps"].mean() - 3 * FDPS_stock["forecast_dps"].std())
        max_mask = FDPS_stock["forecast_dps"] > (FDPS_stock["forecast_dps"].mean() + 3 * FDPS_stock["forecast_dps"].std())
        mask = min_mask | max_mask
        FDPS_stock = FDPS_stock.loc[~mask]
        tempFDPS = FDPS_stock.groupby(['S_INFO_WINDCODE', 'TRADE_DT'])['forecast_dps'].mean().reset_index()
        tempFDPS.index = tempFDPS['TRADE_DT']
        tempFDPS = tempFDPS[['forecast_dps']]
        tempEOD = EOD_stock[['S_DQ_CLOSE']]
        tempEOD.index = EOD_stock['TRADE_DT']
        temp = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"), [tempFDPS, tempEOD])
        temp = temp.sort_index()
        temp.fillna(method='ffill', inplace=True)
        temp['APDTPR'] = temp['forecast_dps'] / temp['S_DQ_CLOSE']
        data_dict[stock] = temp['APDTPR']
    matrix = pd.DataFrame(data_dict).transpose()
    Result = Align(matrix, BASE)
    return Result


# ---------------------------
# Main execution block
# ---------------------------
if __name__ == "__main__":
    # Start and end dates
    start_date = '20050104'
    end_date = '20240524'
    # Output path
    Output_Path = "F:\\因子数据\\描述变量原始值\\"
    # Get the base data: all trading days and tradable stocks
    BASE = getAShareBase(start_date, end_date)
    timelist = gettimelist(BASE)
    tickerlist = gettickerlist(BASE)

    """
    Category: Size/Size/
    Factor Number: 1
    Factor Name: LNCAP: Log of Market Capitalization
    """
    print('Calculating factor LNCAP')
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = LNCAP(BASE, EOD_DI)
    factor.to_csv(Output_Path + 'Size_LNCAP.csv')
    del EOD_DI, factor

    """
    Category: Size/Mid Capitalization/
    Factor Number: 2
    Factor Name: MIDCAP: Cube of Size Exposure
    """
    print('Calculating factor MIDCAP')
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    BASE = getAShareBase(start_date, end_date)
    factor = MIDCAP(BASE, EOD_DI)
    factor.to_csv(Output_Path + 'Mid Capitalization_MIDCAP.csv')
    del EOD_DI, factor

    """
    Category: Volatility/Beta/
    Factor Number: 3
    Factor Name: HBETA: Historical Beta
    """
    print('Calculating factor HBETA')
    EOD = getAShareEODPrices(start_date, end_date)
    INDEX_EOD = getAindexEODPrices(start_date, end_date)
    factor = HBETA(BASE, EOD, INDEX_EOD)
    factor.to_csv(Output_Path + 'Beta_HBETA.csv')

    """
    Category: Volatility/Residual Volatility/
    Factor Number: 4
    Factor Name: HSIGMA: Historical Sigma
    """
    print('Calculating factor HSIGMA')
    EOD = getAShareEODPrices(start_date, end_date)
    INDEX_EOD = getAindexEODPrices(start_date, end_date)
    factor = HSIGMA(BASE, EOD, INDEX_EOD)
    factor.to_csv(Output_Path + 'Residual Volatility_HSIGMA.csv')

    """
    Category: Volatility/Residual Volatility/
    Factor Number: 5
    Factor Name: DASTD: Daily Standard Deviation
    """
    print('Calculating factor DASTD')
    EOD = getAShareEODPrices(start_date, end_date)
    factor = DASTD(BASE, EOD)
    factor.to_csv(Output_Path + 'Residual Volatility_DASTD.csv')
    del EOD, factor

    """
    Category: Volatility/Residual Volatility/
    Factor Number: 6
    Factor Name: CMRA: Cumulative Range
    """
    print('Calculating factor CMRA')
    EOD = getAShareEODPrices(start_date, end_date)
    factor = CMRA(BASE, EOD)
    factor.to_csv(Output_Path + 'Residual Volatility_CMRA.csv')
    del EOD, factor

    """
    Category: Liquidity/Liquidity/
    Factor Number: 7
    Factor Name: STOM: Monthly Share Turnover
    """
    print('Calculating factor STOM')
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = STOM(BASE, EOD_DI)
    factor.to_csv(Output_Path + 'Liquidity_STOM.csv')
    del EOD_DI, factor

    """
    Category: Liquidity/Liquidity/
    Factor Number: 8
    Factor Name: STOQ: Quarterly Share Turnover
    """
    print('Calculating factor STOQ')
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = STOQ(BASE, EOD_DI)
    factor.to_csv(Output_Path + 'Liquidity_STOQ.csv')
    del EOD_DI, factor

    """
    Category: Liquidity/Liquidity/
    Factor Number: 9
    Factor Name: STOA: Annual Share Turnover
    """
    print('Calculating factor STOA')
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = STOA(BASE, EOD_DI)
    factor.to_csv(Output_Path + 'Liquidity_STOA.csv')
    del EOD_DI, factor

    """
    Category: Liquidity/Liquidity/
    Factor Number: 10
    Factor Name: ATVR: Annualized Traded Value Ratio
    """
    print('Calculating factor ATVR')
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = ATVR(BASE, EOD_DI)
    factor.to_csv(Output_Path + 'Liquidity_ATVR.csv')
    del EOD_DI, factor

    """
    Category: Momentum/Short-Term Reversal/
    Factor Number: 11
    Factor Name: STREV: Short-term Reversal
    """
    print('Calculating factor STREV')
    EOD = getAShareEODPrices(start_date, end_date)
    factor = STREV(BASE, EOD)
    factor.to_csv(Output_Path + 'Short-Term Reversal_STREV.csv')
    del EOD, factor

    """
    Category: Momentum/Seasonality/
    Factor Number: 12
    Factor Name: SEASON: Annual Seasonality
    """
    print('Calculating factor SEASON')
    EOD = getAShareEODPrices(start_date, end_date)
    factor = SEASON(BASE, EOD)
    factor.to_csv(Output_Path + 'Seasonality_SEASON.csv')
    del EOD, factor

    """
    Category: Momentum/Industry Momentum/
    Factor Number: 13
    Factor Name: INDMOM: Industry Momentum
    """
    print('Calculating factor INDMOM')
    IndustriesClass = getAShareIndustriesClass()
    AShareCalendar = getAShareCalender()
    I = IndustriesTimeSeries(BASE, IndustriesClass, AShareCalendar)
    EOD = getAShareEODPrices(start_date, end_date)
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = INDMOM(I, EOD, EOD_DI)
    factor.to_csv(Output_Path + 'Industry Momentum_INDMOM.csv')
    del IndustriesClass, AShareCalendar, I, EOD, EOD_DI, factor

    """
    Category: Momentum/Momentum/
    Factor Number: 14
    Factor Name: RSTR: Relative Strength 12-month
    """
    print('Calculating factor RSTR')
    EOD = getAShareEODPrices(start_date, end_date)
    factor = RSTR(BASE, EOD)
    factor.to_csv(Output_Path + 'Momentum_RSTR.csv')
    del EOD, factor

    """
    Category: Momentum/Momentum/
    Factor Number: 15
    Factor Name: HALPHA: Historical Alpha
    """
    print('Calculating factor HALPHA')
    EOD = getAShareEODPrices(start_date, end_date)
    INDEX_EOD = getAindexEODPrices(start_date, end_date)
    factor = HALPHA(BASE, EOD, INDEX_EOD)
    factor.to_csv(Output_Path + 'Momentum_HALPHA.csv')
    del EOD, INDEX_EOD, factor

    """
    Category: Quality/Leverage/
    Factor Number: 16
    Factor Name: MLEV: Market Leverage
    """
    print('Calculating factor MLEV')
    IndustriesClass = getAShareIndustriesClass()
    AShareCalendar = getAShareCalender()
    I = IndustriesTimeSeries(BASE, IndustriesClass, AShareCalendar)
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = MLEV(BASE, I, EOD_DI, BS)
    factor.to_csv(Output_Path + 'Leverage_MLEV.csv')
    del IndustriesClass, AShareCalendar, I, EOD_DI, BS, factor

    """
    Category: Quality/Leverage/
    Factor Number: 17
    Factor Name: BLEV: Book Leverage
    """
    print('Calculating factor BLEV')
    IndustriesClass = getAShareIndustriesClass()
    AShareCalendar = getAShareCalender()
    I = IndustriesTimeSeries(BASE, IndustriesClass, AShareCalendar)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = BLEV(BASE, I, BS)
    factor.to_csv(Output_Path + 'Leverage_BLEV.csv')
    del IndustriesClass, AShareCalendar, I, BS, factor

    """
    Category: Quality/Leverage/
    Factor Number: 18
    Factor Name: DTOA: Debt-to-Assets
    """
    print('Calculating factor DTOA')
    IndustriesClass = getAShareIndustriesClass()
    AShareCalendar = getAShareCalender()
    I = IndustriesTimeSeries(BASE, IndustriesClass, AShareCalendar)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = DTOA(BASE, I, BS)
    factor.to_csv(Output_Path + 'Leverage_DTOA.csv')
    del IndustriesClass, AShareCalendar, I, BS, factor

    """
    Category: Quality/Earnings Variability/
    Factor Number: 19
    Factor Name: VSAL: Variability in Sales
    """
    print('Calculating factor VSAL')
    INCOME = getAShareIncome(start_date, end_date)
    factor = VSAL(BASE, INCOME)
    factor.to_csv(Output_Path + 'Earnings Variability_VSAL.csv')
    del INCOME, factor

    """
    Category: Quality/Earnings Variability/
    Factor Number: 20
    Factor Name: VERN: Variability in Earnings
    """
    print('Calculating factor VERN')
    INCOME = getAShareIncome(start_date, end_date)
    factor = VERN(BASE, INCOME)
    factor.to_csv(Output_Path + 'Earnings Variability_VERN.csv')
    del INCOME, factor

    """
    Category: Quality/Earnings Variability/
    Factor Number: 21
    Factor Name: VFLO: Variability in Cash-flows
    """
    print('Calculating factor VFLO')
    CF = getAShareCashFlow(start_date, end_date)
    factor = VFLO(BASE, CF)
    factor.to_csv(Output_Path + 'Earnings Variability_VFLO.csv')
    del CF, factor

    """
    Category: Quality/Earnings Variability/
    Factor Number: 22
    Factor Name: ETOPF_STD: Standard deviation of Analyst Forecast Earnings-to-Price
    """
    print('Calculating factor ETOPF_STD')
    tickerlist = gettickerlist(BASE)
    FEPS = getForecastEPS(start_date, end_date, tickerlist)
    EOD = getAShareEODPrices(start_date, end_date)
    factor = ETOPF_STD(BASE, FEPS, EOD)
    factor.to_csv(Output_Path + 'Earnings Variability_ETOPF_STD.csv')
    del FEPS, EOD, factor

    """
    Category: Quality/Earnings Quality/
    Factor Number: 23
    Factor Name: ABS: Accruals - Balance Sheet Version
    """
    print('Calculating factor ABS')
    FI = getAShareFinancialIndicator(start_date, end_date)
    CF = getAShareCashFlow(start_date, end_date)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = ABS(BASE, FI, CF, BS)
    factor.to_csv(Output_Path + 'Earnings Quality_ABS.csv')
    del FI, CF, BS, factor

    """
    Category: Quality/Earnings Quality/
    Factor Number: 24
    Factor Name: ACF: Accruals - Cashflow Statement Version
    """
    print('Calculating factor ACF')
    FI = getAShareFinancialIndicator(start_date, end_date)
    CF = getAShareCashFlow(start_date, end_date)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = ACF(BASE, FI, CF, BS)
    factor.to_csv(Output_Path + 'Earnings Quality_ACF.csv')
    del FI, CF, BS, factor

    """
    Category: Quality/Profitability/
    Factor Number: 25
    Factor Name: ATO: Asset Turnover
    """
    print('Calculating factor ATO')
    TTM = getAShareTTMHis(start_date, end_date)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = ATO(BASE, TTM, BS)
    factor.to_csv(Output_Path + 'Profitability_ATO.csv')
    del TTM, BS, factor

    """
    Category: Quality/Profitability/
    Factor Number: 26
    Factor Name: GP: Gross Profitability
    """
    print('Calculating factor GP')
    INCOME = getAShareIncome(start_date, end_date)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = GP(BASE, INCOME, BS)
    factor.to_csv(Output_Path + 'Profitability_GP.csv')
    del INCOME, BS, factor

    """
    Category: Quality/Profitability/
    Factor Number: 27
    Factor Name: GPM: Gross Profit Margin
    """
    print('Calculating factor GPM')
    INCOME = getAShareIncome(start_date, end_date)
    factor = GPM(BASE, INCOME)
    factor.to_csv(Output_Path + 'Profitability_GPM.csv')
    del INCOME, factor

    """
    Category: Quality/Profitability/
    Factor Number: 28
    Factor Name: ROA: Return on Assets
    """
    print('Calculating factor ROA')
    TTM = getAShareTTMHis(start_date, end_date)
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = ROA(BASE, TTM, BS)
    factor.to_csv(Output_Path + 'Profitability_ROA.csv')
    del TTM, BS, factor

    """
    Category: Quality/Investment Quality/
    Factor Number: 29
    Factor Name: AGRO: Total Assets Growth Rate
    """
    print('Calculating factor AGRO')
    BS = getAShareBalanceSheet(start_date, end_date)
    factor = AGRO(BASE, BS)
    factor.to_csv(Output_Path + 'Investment Quality_AGRO.csv')
    del BS, factor

    """
    Category: Quality/Investment Quality/
    Factor Number: 30
    Factor Name: IGRO: Issuance Growth
    """
    print('Calculating factor IGRO')
    CAP = getAShareCapitalization(start_date, end_date)
    TC = getAShareTypeCode()
    CL = getAShareCalender()
    factor = IGRO(BASE, CAP, TC, CL)
    factor.to_csv(Output_Path + 'Investment Quality_IGRO.csv')
    del CAP, TC, factor

    """
    Category: Quality/Investment Quality/
    Factor Number: 31
    Factor Name: CXGRO: Capital Expenditure Growth
    """
    print('Calculating factor CXGRO')
    CF = getAShareCashFlow(start_date, end_date)
    factor = CXGRO(BASE, CF)
    factor.to_csv(Output_Path + 'Investment Quality_CXGRO.csv')
    del CF, factor

    """
    Category: Value/Book-to-Price/
    Factor Number: 32
    Factor Name: BTOP: Book-to-Price
    """
    print('Calculating factor BTOP')
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = BTOP(BASE, EOD_DI)
    factor.to_csv(Output_Path + 'Book-to-Price_BTOP.csv')
    del EOD_DI, factor

    """
    Category: Value/Earnings Yield/
    Factor Number: 33
    Factor Name: ETOP: Earnings-to-Price
    """
    print('Calculating factor ETOP')
    TTM = getAShareTTMHis(start_date, end_date)
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = ETOP(BASE, TTM, EOD_DI)
    factor.to_csv(Output_Path + 'Earnings Yield_ETOP.csv')
    del EOD_DI, factor

    """
    Category: Value/Earnings Yield/
    Factor Number: 34
    Factor Name: ETOPF: Analyst-Predicted Earnings-to-Price
    """
    print('Calculating factor ETOPF')
    FEP = getForecastEP(start_date, end_date, tickerlist)
    factor = ETOPF(BASE, FEP)
    factor.to_csv(Output_Path + 'Earnings Yield_ETOPF.csv')
    del FEP, factor

    """
    Category: Value/Earnings Yield/
    Factor Number: 35
    Factor Name: CETOP: Cash-Earnings-to-Price
    """
    print('Calculating factor CETOP')
    TTM = getAShareTTMHis(start_date, end_date)
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    factor = CETOP(BASE, TTM, EOD_DI)
    factor.to_csv(Output_Path + 'Earnings Yield_CETOP.csv')
    del TTM, EOD_DI, factor

    """
    Category: Value/Earnings Yield/
    Factor Number: 36
    Factor Name: EM: Enterprise Multiple (EBIT to EV)
    """
    print('Calculating factor EM')
    INCOME = getAShareIncome(start_date, end_date)
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    BS = getAShareBalanceSheet(start_date, end_date)
    CF = getAShareCashFlow(start_date, end_date)
    factor = EM(BASE, INCOME, EOD_DI, BS, CF)
    factor.to_csv(Output_Path + 'Earnings Yield_EM.csv')
    del INCOME, EOD_DI, BS, CF, factor

    """
    Category: Value/Long-Term Reversal/
    Factor Number: 37
    Factor Name: LTRSTR: Long-term Relative Strength
    """
    print('Calculating factor LTRSTR')
    EOD = getAShareEODPrices(start_date, end_date)
    factor = LTRSTR(BASE, EOD)
    factor.to_csv(Output_Path + 'Long-Term Reversal_LTRSTR.csv')
    del EOD, factor

    """
    Category: Value/Long-Term Reversal/
    Factor Number: 38
    Factor Name: LTHALPHA: Long-term Historical Alpha
    """
    print('Calculating factor LTHALPHA')
    EOD = getAShareEODPrices(start_date, end_date)
    INDEX_EOD = getAindexEODPrices(start_date, end_date)
    factor = LTHALPHA(BASE, EOD, INDEX_EOD)
    factor.to_csv(Output_Path + 'Long-Term Reversal_LTHALPHA.csv')
    del EOD, INDEX_EOD, factor

    """
    Category: Growth/Growth/
    Factor Number: 39
    Factor Name: EGRLF: Analyst Predicted Earnings Long-term Growth
    """
    print('Calculating factor EGRLF')
    NP_FY1 = getforecastNP_FY1(start_date, end_date, tickerlist)
    NP_FY2 = getforecastNP_FY2(start_date, end_date, tickerlist)
    NP_FY3 = getforecastNP_FY3(start_date, end_date, tickerlist)
    factor = EGRLF(BASE, NP_FY1, NP_FY2, NP_FY3)
    factor.to_csv(Output_Path + 'Growth_EGRLF.csv')
    del NP_FY1, NP_FY2, NP_FY3, factor

    """
    Category: Growth/Growth/
    Factor Number: 40
    Factor Name: EGRO: Earnings per Share Growth Rate
    """
    print('Calculating factor EGRO')
    INCOME = getAShareIncome(start_date, end_date)
    factor = EGRO(BASE, INCOME)
    factor.to_csv(Output_Path + 'Growth_EGRO.csv')
    del INCOME, factor

    """
    Category: Growth/Growth/
    Factor Number: 41
    Factor Name: SGRO: Sales per Share Growth Rate
    """
    print('Calculating factor SGRO')
    INCOME = getAShareIncome(start_date, end_date)
    factor = SGRO(BASE, INCOME)
    factor.to_csv(Output_Path + 'Growth_SGRO.csv')
    del INCOME, factor

    """
    Category: Sentiment/Analyst Sentiment/
    Factor Number: 42
    Factor Name: RR: Revision Ratio
    """
    print('Calculating factor RR')
    tickerlist = gettickerlist(BASE)
    FEA = getForecastEarningsAdjust(start_date, end_date, tickerlist)
    factor = RR(BASE, FEA)
    factor.to_csv(Output_Path + 'Analyst Sentiment_RR.csv')
    del FEA, factor

    """
    Category: Sentiment/Analyst Sentiment/
    Factor Number: 43
    Factor Name: ETOPF_C: Change in Analyst-Predicted Earnings-to-Price
    """
    print('Calculating factor ETOPF_C')
    tickerlist = gettickerlist(BASE)
    FEP = getForecastEP(start_date, end_date, tickerlist)
    factor = ETOPF_C(BASE, FEP)
    factor.to_csv(Output_Path + 'Analyst Sentiment_ETOPF_C.csv')
    del FEP, factor

    """
    Category: Sentiment/Analyst Sentiment/
    Factor Number: 44
    Factor Name: EPSF_C: Change in Analyst-Predicted Earnings per Share
    """
    print('Calculating factor EPSF_C')
    tickerlist = gettickerlist(BASE)
    FEPS = getForecastEPS(start_date, end_date, tickerlist)
    factor = EPSF_C(BASE, FEPS)
    factor.to_csv(Output_Path + 'Analyst Sentiment_EPSF_C.csv')
    del FEPS, factor

    start = time.perf_counter()
    """
    Category: Dividend Yield/Dividend Yield/
    Factor Number: 45
    Factor Name: DTOP: Dividend-to-Price
    """
    print('Calculating factor DTOP')
    EOD = getAShareEODPrices(start_date, end_date)
    DVD = getAShareDividend(start_date, end_date)
    INCOME = getAShareIncome(start_date, end_date)
    AllPeriod = getAllPeriod(INCOME)
    factor = DTOP(BASE, AllPeriod, EOD, DVD)
    factor.to_csv(Output_Path + 'Dividend Yield_DTOP.csv')
    del EOD, DVD, AllPeriod, factor

    """
    Category: Dividend Yield/Dividend Yield/
    Factor Number: 46
    Factor Name: DTOPF: Analyst-Predicted Dividend-to-Price
    """
    print('Calculating factor DTOPF')
    EOD = getAShareEODPrices(start_date, end_date)
    tickerlist = gettickerlist(BASE)
    FDPS = getForecastDPS(start_date, end_date, tickerlist)
    factor = DTOPF(BASE, FDPS, EOD)
    factor.to_csv(Output_Path + 'Dividend Yield_DTOPF.csv')
    del EOD, FDPS, factor
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")
