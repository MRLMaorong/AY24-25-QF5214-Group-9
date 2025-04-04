#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:29:55 2025

@author: hejundong
"""

import pandas as pd

df1 = pd.read_excel('esg平均评级因子和评级分歧因子.xls')
df2 = pd.read_csv('调整排序后月频回归的面板数据.csv')

df1['TRADE_DT'] = df1['year'] + 1
df1 = df1[['stock', 'TRADE_DT', 'ESG_uncertainty', 'ESG_rank']]
df1 = df1.fillna(0)


df2['stock'] = df2['S_INFO_WINDCODE'].apply(lambda x: int(x[:6]))

df2['TRADE_DT'] = df2['TRADE_DT']//10000

merged_df = pd.merge(df2, df1, on=['stock', 'TRADE_DT'], how='inner')

#%%
import statsmodels.api as sm

# 假设 merged_df 已经包含你需要的所有数据
# 首先，确保相关列没有缺失值
df = merged_df.dropna(subset=['next_Rtn', 'ESG_uncertainty', 'ESG_rank'])

# 定义自变量和因变量
X = df[['ESG_uncertainty', 'ESG_rank']]
y = df['next_Rtn']

# 为自变量添加常数项（截距）
X = sm.add_constant(X)

# 建立并拟合线性回归模型
model = sm.OLS(y, X).fit()

# 输出模型摘要，查看各变量的系数和 p 值
print(model.summary())

#%%
import pandas as pd
import statsmodels.api as sm

# 定义变量列表：基本模型的自变量（排除 ESG 相关变量）
base_columns = [
    'Analyst Sentiment', 'Beta', 'Book-to-Price', 'Dividend Yield',
    'Earnings Quality', 'Earnings Variability', 'Earnings Yield', 'Growth',
    'Industry Momentum', 'Investment Quality', 'Leverage', 'Liquidity',
    'Long-Term Reversal', 'Mid Capitalization', 'Momentum', 'Profitability',
    'Residual Volatility', 'Seasonality', 'Short-Term Reversal', 'Size'
]

# ESG 变量
esg_columns = ['ESG_uncertainty', 'ESG_rank']

# 因变量
dependent = 'next_Rtn'

# ============================
# 基本模型：只使用金融因子
# ============================
# 删除因变量和基本自变量的缺失值
model1_df = merged_df.dropna(subset=[dependent] + base_columns)
X1 = model1_df[base_columns]
y1 = model1_df[dependent]

# 加入常数项
X1 = sm.add_constant(X1)

# 拟合基本模型
model1 = sm.OLS(y1, X1).fit()
print("基本模型结果：")
print(model1.summary())

# ============================
# 扩展模型：在基本模型上加入 ESG 变量
# ============================
# 为保证样本一致，删除因变量、基本自变量和 ESG 变量缺失值
model2_df = merged_df.dropna(subset=[dependent] + base_columns + esg_columns)
X2 = model2_df[base_columns + esg_columns]
y2 = model2_df[dependent]

# 加入常数项
X2 = sm.add_constant(X2)

# 拟合扩展模型
model2 = sm.OLS(y2, X2).fit()
print("\n扩展模型结果（含 ESG 变量）：")
print(model2.summary())

# ============================
# 进行嵌套模型的 F 检验
# ============================
# 为保证两模型基于相同样本，使用 model2_df 来构建基本模型的嵌套版本
X1_nested = model2_df[base_columns]
y_nested = model2_df[dependent]
X1_nested = sm.add_constant(X1_nested)
nested_model1 = sm.OLS(y_nested, X1_nested).fit()

# 通过 compare_f_test 比较扩展模型和嵌套基本模型
f_test_result = model2.compare_f_test(nested_model1)
print("\nF 检验结果（扩展模型 vs 基本模型）：")
print("F统计量：", f_test_result[0])
print("p值：", f_test_result[1])
print("自由度差：", f_test_result[2])