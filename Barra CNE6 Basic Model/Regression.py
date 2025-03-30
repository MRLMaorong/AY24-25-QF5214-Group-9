# -*- coding: utf-8 -*-
"""
Created in 2025

Function: Factor Regression Program

"""

import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings(action='ignore')
from tqdm import tqdm
import pickle
# from LoadSQL import *
from Tools import *
import statsmodels.sandbox.rls as rls
# Enable Chinese display
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
pd.set_option('display.encoding', 'gbk')
from BackTest import *


if __name__ == "__main__":
    # Factors start from 20050101
    factor_start_date = '20050104'
    factor_end_date = '20240524'
    # Regression starts from 20100601
    reg_start_date = '20100601'
    reg_end_date = '20240524'
    # For the comprehensive financial industry, component stocks become available from 20191202;
    # therefore, factor exposures start from 20191202 and regression needs to be done in segments.
    mid_date = '20191202'

    # Regression frequency
    frequence  = '月频'  # "Monthly"
    # Factor type
    for type in ['风格因子','CNLT大类因子']:  # "Style Factors" and "CNLT First-Level Factors"
        # Set input/output paths
        path_factor = "F:/因子数据/"  # "Factor Data" folder
        path_input = "F:/因子数据/Input/"  # Input folder (contains files with Chinese names)
        path_w = "F:/因子数据/因子回归和纯因子组合/" + type + "因子回归结果/"  # "Factor Regression and Pure Factor Combination" results folder for the given type
        if not os.path.exists(path_w):
            os.makedirs(path_w)
        # Define factor names
        if type == 'CNLT大类因子':
            style_factor_names = ['GROWTH','LIQUIDITY','MOMENTUM','QUALITY','SIZE','VALUE','VOLATILITY','YIELD']
        elif type == '风格因子':
            style_factor_names = ["Analyst Sentiment","Beta","Book-to-Price","Dividend Yield","Earnings Quality","Earnings Variability",
                                  "Earnings Yield","Growth","Industry Momentum","Investment Quality",
                                  "Leverage","Liquidity","Long-Term Reversal","Mid Capitalization","Momentum",
                                  "Profitability","Residual Volatility","Seasonality","Short-Term Reversal","Size"]

        industry_factor_names_old = ['CI005001', 'CI005002', 'CI005003', 'CI005004', 'CI005005', 'CI005006',
                                      'CI005007', 'CI005008', 'CI005009', 'CI005010', 'CI005011', 'CI005012',
                                      'CI005013', 'CI005014', 'CI005015', 'CI005016', 'CI005017', 'CI005018',
                                      'CI005019', 'CI005020', 'CI005021', 'CI005022', 'CI005023', 'CI005024',
                                      'CI005025', 'CI005026', 'CI005027', 'CI005028', 'CI005029']
        industry_factor_names_new = ['CI005001', 'CI005002', 'CI005003', 'CI005004', 'CI005005', 'CI005006',
                                      'CI005007', 'CI005008', 'CI005009', 'CI005010', 'CI005011', 'CI005012',
                                      'CI005013', 'CI005014', 'CI005015', 'CI005016', 'CI005017', 'CI005018',
                                      'CI005019', 'CI005020', 'CI005021', 'CI005022', 'CI005023', 'CI005024',
                                      'CI005025', 'CI005026', 'CI005027', 'CI005028', 'CI005029', 'CI005030']
        country_factor_names = ['Country']

        # Build the regression BASE data (including basic data and factor values)
        print('Reading data...')
        # Get all trading days and tradable stocks as the base
        BASE = getAShareBase(factor_start_date, factor_end_date)
        # Merge basic data including listing date, free float market cap, adjusted close price, and volume
        LD = getAShareListDate()
        factor = pd.merge(BASE, LD, how='left', on='S_INFO_WINDCODE')
        del LD
        EOD_DI = getAShareEODDerivativeIndicator(factor_start_date, factor_end_date)[['S_INFO_WINDCODE', 'TRADE_DT','FREE_FLOAT_CAP']]
        factor = pd.merge(factor, EOD_DI, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])
        del EOD_DI
        EOD = getAShareEODPrices(factor_start_date, factor_end_date)[['S_INFO_WINDCODE', 'TRADE_DT','S_DQ_ADJCLOSE','S_DQ_VOLUME']]
        factor = pd.merge(factor, EOD, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])
        del EOD
        factor = factor[['S_INFO_WINDCODE','TRADE_DT','S_INFO_LISTDATE','FREE_FLOAT_CAP', 'S_DQ_ADJCLOSE','S_DQ_VOLUME']]

        Calender_file = path_input + '交易日.xlsx'  # "交易日.xlsx" means "Trading Days.xlsx"
        Calender = Load_Calender(Calender_file, freq='月频')
        Calender = Calender[(Calender['TRADE_DT'] >= reg_start_date) & (Calender['TRADE_DT'] <= reg_end_date)]

        # Read all style, industry, and country factors
        for factor_name in tqdm(country_factor_names + industry_factor_names_new + style_factor_names):
            if factor_name in country_factor_names + industry_factor_names_new:
                factor_tmp = pd.read_csv(path_factor + "行业和国家因子/" + factor_name + ".csv", index_col=0)  # "行业和国家因子" means "Industry and Country Factors"
            elif factor_name in style_factor_names:
                factor_tmp = pd.read_csv(path_factor + type + "/" + factor_name + ".csv", index_col=0)  # "风格因子" or "CNLT大类因子" folder
            factor_tmp_df = factor_format_adjust(factor_tmp, factor_name)
            factor = pd.merge(factor, factor_tmp_df, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])
        # Save the merged factor data as a pickle file for later use.
        factor.reset_index(inplace=True, drop=True)
        factor = factor[['S_INFO_WINDCODE', 'TRADE_DT', 'S_INFO_LISTDATE', 'FREE_FLOAT_CAP', 'S_DQ_ADJCLOSE', 'S_DQ_VOLUME']
                        + country_factor_names + industry_factor_names_new + style_factor_names]
        if type == 'CNLT大类因子':
            with open(path_factor + "factor_first.pkl", "wb") as file:
                pickle.dump(factor, file)
        elif type == '风格因子':
            with open(path_factor + "factor_second.pkl", "wb") as file:
                pickle.dump(factor, file)

        # Begin the regression part
        if type == 'CNLT大类因子':
            with open(path_factor + "factor_first.pkl", "rb") as file:
                factor = pickle.load(file)
        elif type == '风格因子':
            with open(path_factor + "factor_second.pkl", "rb") as file:
                factor = pickle.load(file)
        factor = factor[factor['TRADE_DT'] >= '20100601']

        # Filter data based on the specified frequency
        if frequence == '月频':  # "月频" means "Monthly"
            datelist = pd.read_excel(path_input + "交易日.xlsx", sheet_name='月频')  # "交易日.xlsx" here is the trading days file for monthly frequency.
            datelist['调仓日期'] = pd.to_datetime([str(i) for i in datelist['调仓日期']]).strftime('%Y%m%d')  # "调仓日期" means "rebalancing dates"
            datelist.rename(columns={'调仓日期': 'TRADE_DT'}, inplace=True)
            factor = pd.merge(factor, datelist, how='inner', on='TRADE_DT')
            factor.reset_index(inplace=True, drop=True)

        # Get the next period date, denoted as T (for regression, factor exposure is T-1 and stock return is T)
        factor.rename(columns={'TRADE_DT': 'T-1'}, inplace=True)
        factor = factor.groupby(factor["S_INFO_WINDCODE"]).apply(next_date).reset_index(drop=True)
        # Calculate T-period returns
        factor = factor.groupby(factor["S_INFO_WINDCODE"]).apply(next_rtn).reset_index(drop=True)
        # Filter regression data by the specified regression dates
        factor = factor[(factor['T-1'] >= reg_start_date) & (factor['T-1'] <= reg_end_date)]
        print(factor.columns)

        # Filter the stock pool: remove stocks listed for less than 6 months.
        factor = factor[(pd.to_datetime(factor['T-1']) > (pd.to_datetime(factor['S_INFO_LISTDATE']) + datetime.timedelta(days=180)))]
        # Determine if stocks are trading normally.
        factor = factor.groupby(factor["S_INFO_WINDCODE"]).apply(trading).reset_index(drop=True)
        print(factor.columns)
        # Remove stocks that are not trading.
        factor = factor[factor['trading'] == True]
        # In very rare cases, free float market cap data is missing (e.g., 300847.SZ on 20220426, 300720.SZ and 300886.SZ on 20220428);
        # fill missing values with the previous value.
        factor['FREE_FLOAT_CAP'] = factor.groupby('S_INFO_WINDCODE')['FREE_FLOAT_CAP'].fillna(method='ffill')
        factor = factor.sort_values(by=['T', 'S_INFO_WINDCODE']).reset_index(drop=True)

        # Perform WLS regression to calculate regression coefficients f, T-values, P-values, R2, and residuals.
        f = pd.DataFrame(index=factor['T'].unique(), columns=country_factor_names + industry_factor_names_new + style_factor_names)
        tvalue = pd.DataFrame(index=factor['T'].unique(), columns=country_factor_names + industry_factor_names_new + style_factor_names)
        pvalue = pd.DataFrame(index=factor['T'].unique(), columns=country_factor_names + industry_factor_names_new + style_factor_names)
        R2 = pd.DataFrame(index=factor['T'].unique(), columns=['R2'])
        residual = factor[['S_INFO_WINDCODE', 'T']]
        residual['residual'] = np.nan
        Pred_Return = factor[['S_INFO_WINDCODE', 'T']]
        Pred_Return['pred_return'] = np.nan
        # Pure factor combination result
        Portfolio = pd.DataFrame(columns=['TRADE_DT', 'S_INFO_WINDCODE'] + country_factor_names + industry_factor_names_new + style_factor_names)
        Target_Exposure = pd.DataFrame(index=factor['T-1'].unique(), columns=country_factor_names + industry_factor_names_new + style_factor_names)
        
        def WLS(df):
            df.reset_index(drop=True, inplace=True)
            T = list(df["T"])[0]  # T corresponds to the return date (Y)
            T_1 = list(df["T-1"])[0]  # T-1 corresponds to the factor exposure date (X)
            # Fill missing factor exposures with the market median if needed.
            df[style_factor_names] = df[style_factor_names].fillna(df[style_factor_names].median(axis=0))
            # Segment the regression based on mid_date
            if T_1 < mid_date:  # For T-1 before mid_date, use the old industry factor names.
                stocklist = df['S_INFO_WINDCODE'].values
                X = df[country_factor_names + industry_factor_names_old + style_factor_names].values
                Y = df['next_Rtn'].values
                stockCap = df['FREE_FLOAT_CAP'].values
                stockCap_weight = np.sqrt(stockCap) / np.sqrt(stockCap).sum()
                # Compute industry free float market cap weights
                indCap = [df[df[ind_name] == 1]['FREE_FLOAT_CAP'].sum() for ind_name in industry_factor_names_old]
                indCap_weight = indCap / np.sum(indCap)
                A = np.concatenate((np.zeros(len(country_factor_names)), indCap_weight, np.zeros(len(style_factor_names))), axis=0)
                b = 0
                model = rls.RLS(Y, X, constr=A, param=b, sigma=stockCap_weight).fit()
                f.loc[T, country_factor_names + industry_factor_names_old + style_factor_names] = model.params
                residual.loc[residual['T'] == T, 'residual'] = model.resid
                Pred_Return.loc[Pred_Return['T'] == T, 'pred_return'] = model.predict(X)
                tvalue.loc[T, country_factor_names + industry_factor_names_old + style_factor_names] = model.tvalues
                pvalue.loc[T, country_factor_names + industry_factor_names_old + style_factor_names] = model.pvalues
                R2.loc[T, 'R2'] = 1 - np.nansum(np.power(model.resid, 2) * stockCap_weight) / np.nansum(np.power(Y, 2) * stockCap_weight)
                
                # Pure factor combination
                # Construct the constraint matrix C for least squares; dimensions (1+I+S, I+S). See Equation 7.26 in Ishikawa's "Factor Investing".
                C = np.eye(len(country_factor_names) + len(industry_factor_names_old) + len(style_factor_names) - 1)
                insert_array = np.concatenate((np.zeros(len(country_factor_names)), 
                                                 np.array([-indCap_weight[i] / indCap_weight[-1] for i in range(len(industry_factor_names_old) - 1)]),
                                                 np.zeros(len(style_factor_names))), axis=0)
                C = np.insert(arr=C, obj=len(country_factor_names) + len(industry_factor_names_old) - 1, values=insert_array, axis=0)
                W = np.diag(stockCap_weight)
                Omega = C @ np.linalg.inv(C.T @ X.T @ W @ X @ C) @ C.T @ X.T @ W
                Portfolio_today = pd.DataFrame(index=stocklist, columns=country_factor_names + industry_factor_names_old + style_factor_names, data=Omega.T)
                Portfolio_today.reset_index(inplace=True)
                Portfolio_today.rename(columns={"index": "S_INFO_WINDCODE"}, inplace=True)
                Portfolio_today.insert(loc=0, column='TRADE_DT', value=T_1)
                Target_Exposure.loc[T_1, country_factor_names + industry_factor_names_old + style_factor_names] = np.diag(Omega @ X)

            elif T_1 >= mid_date:  # Use new industry factor names for T-1 after mid_date.
                stocklist = df['S_INFO_WINDCODE'].values
                X = df[country_factor_names + industry_factor_names_new + style_factor_names].values
                Y = df['next_Rtn'].values
                stockCap = df['FREE_FLOAT_CAP'].values
                stockCap_weight = np.sqrt(stockCap) / np.sqrt(stockCap).sum()
                indCap = [df[df[ind_name] == 1]['FREE_FLOAT_CAP'].sum() for ind_name in industry_factor_names_new]
                indCap_weight = indCap / np.sum(indCap)
                A = np.concatenate((np.zeros(len(country_factor_names)), indCap_weight, np.zeros(len(style_factor_names))), axis=0)
                b = 0
                model = rls.RLS(Y, X, constr=A, param=b, sigma=stockCap_weight).fit()
                f.loc[T, country_factor_names + industry_factor_names_new + style_factor_names] = model.params
                residual.loc[residual['T'] == T, 'residual'] = model.resid
                Pred_Return.loc[Pred_Return['T'] == T, 'pred_return'] = model.predict(X)
                tvalue.loc[T, country_factor_names + industry_factor_names_new + style_factor_names] = model.tvalues
                pvalue.loc[T, country_factor_names + industry_factor_names_new + style_factor_names] = model.pvalues
                R2.loc[T, 'R2'] = 1 - np.nansum(np.power(model.resid, 2) * stockCap_weight) / np.nansum(np.power(Y, 2) * stockCap_weight)

                C = np.eye(len(country_factor_names) + len(industry_factor_names_new) + len(style_factor_names) - 1)
                insert_array = np.concatenate((np.zeros(len(country_factor_names)), 
                                                 np.array([-indCap_weight[i] / indCap_weight[-1] for i in range(len(industry_factor_names_new) - 1)]),
                                                 np.zeros(len(style_factor_names))), axis=0)
                C = np.insert(arr=C, obj=len(country_factor_names) + len(industry_factor_names_new) - 1, values=insert_array, axis=0)
                W = np.diag(stockCap_weight)
                Omega = C @ np.linalg.inv(C.T @ X.T @ W @ X @ C) @ C.T @ X.T @ W
                Portfolio_today = pd.DataFrame(index=stocklist, columns=country_factor_names + industry_factor_names_new + style_factor_names, data=Omega.T)
                Portfolio_today.reset_index(inplace=True)
                Portfolio_today.rename(columns={"index": "S_INFO_WINDCODE"}, inplace=True)
                Portfolio_today.insert(loc=0, column='TRADE_DT', value=T_1)
                Target_Exposure.loc[T_1, country_factor_names + industry_factor_names_new + style_factor_names] = np.diag(Omega @ X)
            return Portfolio_today

        # Loop over cross-sectional regressions to obtain regression coefficients and residuals.
        tqdm.pandas(desc="Looping over cross-sectional regressions")
        Portfolio = factor.groupby('T').progress_apply(WLS).reset_index(drop=True)
        # Save the results
        writer = pd.ExcelWriter(path_w + frequence + "回归结果.xlsx")  # "回归结果.xlsx" means "Regression Results.xlsx"
        f.to_excel(writer, sheet_name='因子收益率')  # "因子收益率" means "Factor Returns"
        np.cumprod(1 + f).to_excel(writer, sheet_name='因子净值')  # "因子净值" means "Factor Net Value"
        tvalue.to_excel(writer, sheet_name='T值')  # "T值" means "T-Values"
        pvalue.to_excel(writer, sheet_name='P值')  # "P值" means "P-Values"
        R2.to_excel(writer, sheet_name='R方')  # "R方" means "R-squared"
        writer._save()
        Portfolio.to_csv(path_w + frequence + "纯因子组合.csv", index=False)  # "纯因子组合.csv" means "Pure Factor Combination.csv"
        Target_Exposure.to_csv(path_w + frequence + "目标暴露优化结果.csv")  # "目标暴露优化结果.csv" means "Target Exposure Optimization Results.csv"
        residual.to_csv(path_w + frequence + "回归残差.csv", index=False)  # "回归残差.csv" means "Regression Residuals.csv"
        Pred_Return.to_csv(path_w + frequence + "回归收益预测.csv", index=False)  # "回归收益预测.csv" means "Regression Return Predictions.csv"

        print("Saving regression panel data...")
        factor_panel = factor[['T-1', 'S_INFO_WINDCODE'] + country_factor_names + industry_factor_names_new + style_factor_names + ['next_Rtn']]
        factor_panel.rename(columns={'T-1': 'TRADE_DT'}, inplace=True)
        factor_panel.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)

        # Output regression panel data
        factor_panel.to_csv(path_w + frequence + "_回归面板数据.csv", index=False)  # "_回归面板数据.csv" means "_Regression Panel Data.csv"
        print("Regression panel data saved:", path_w + frequence + "_回归面板数据.csv")

