# -*- coding: utf-8 -*-
"""
Created in 2025

Functionality: Factor Processing and Weighting Program

"""

import warnings

warnings.filterwarnings(action='ignore')
from LoadSQL import *
from Tools import *
import pickle
import statsmodels.api as sm
# Enable Chinese display
import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
pd.set_option('display.encoding', 'gbk')


def GetProcessData(start_date, end_date, BASE):
    '''
    Read data used for factor processing and weighting.
    '''
    print('Reading data for factor processing and weighting')
    # Read free float market capitalization, which is used as the weighting for factor standardization.
    EOD_DI = getAShareEODDerivativeIndicator(start_date, end_date)
    weight_df = FreeFloatCap_Weight_df(BASE, EOD_DI)
    # Get industry information for filling missing values with industry medians.
    IndustriesClass = getAShareIndustriesClass()
    AShareCalendar = getAShareCalender()
    I = IndustriesTimeSeries(BASE, IndustriesClass, AShareCalendar)
    return weight_df, I


def CreatIndustryCountry(BASE, I, path):
    '''
    Generate industry and country factors.
    '''
    print('Generating industry and country factors')
    # Create dummy variables for industries.
    I = I.join(pd.get_dummies(I['INDUSTRIESNAME']))
    I.drop('INDUSTRIESNAME', axis=1, inplace=True)

    industry_factor_names = ['CI005001', 'CI005002', 'CI005003', 'CI005004', 'CI005005', 'CI005006',
                             'CI005007', 'CI005008', 'CI005009', 'CI005010', 'CI005011', 'CI005012',
                             'CI005013', 'CI005014', 'CI005015', 'CI005016', 'CI005017', 'CI005018',
                             'CI005019', 'CI005020', 'CI005021', 'CI005022', 'CI005023', 'CI005024',
                             'CI005025', 'CI005026', 'CI005027', 'CI005028', 'CI005029', 'CI005030']

    industry_dict = {'CI005001': '石油石化', 'CI005002': '煤炭', 'CI005003': '有色金属', 'CI005004': '电力及公用事业', 'CI005005': '钢铁', 'CI005006': '基础化工',
                     'CI005007': '建筑', 'CI005008': '建材', 'CI005009': '轻工制造', 'CI005010': '机械', 'CI005011': '电力设备及新能源', 'CI005012': '国防军工',
                     'CI005013': '汽车', 'CI005014': '商贸零售', 'CI005015': '消费者服务', 'CI005016': '家电', 'CI005017': '纺织服装', 'CI005018': '医药',
                     'CI005019': '食品饮料', 'CI005020': '农林牧渔', 'CI005021': '银行', 'CI005022': '非银行金融', 'CI005023': '房地产', 'CI005024': '交通运输',
                     'CI005025': '电子', 'CI005026': '通信', 'CI005027': '计算机', 'CI005028': '传媒', 'CI005029': '综合', 'CI005030': '综合金融'}
    # Convert industry factor values from True/False to 1/0.
    for factor_name in industry_factor_names:
        I[industry_dict[factor_name]] = I[industry_dict[factor_name]].astype(np.float64)
    # Save factors to CSV.
    if not os.path.exists(path + "行业和国家因子\\"):
        os.makedirs(path + "行业和国家因子\\")
    for factor_name in tqdm(industry_factor_names):
        factor = I[['S_INFO_WINDCODE', 'TRADE_DT', industry_dict[factor_name]]]
        matrix = pd.pivot(factor, index='S_INFO_WINDCODE', values=industry_dict[factor_name], columns='TRADE_DT')
        # Align with the base.
        Result = Align(matrix, BASE)
        Result.to_csv(path + "行业和国家因子\\" + factor_name + ".csv")
    # Add country factor.
    I.insert(loc=len(I.columns), column='Country', value=1.00)
    country_factor_names = ['Country']
    for factor_name in country_factor_names:
        factor = I[['S_INFO_WINDCODE', 'TRADE_DT', factor_name]]
        matrix = pd.pivot(factor, index='S_INFO_WINDCODE', values=factor_name, columns='TRADE_DT')
        # Align with the base.
        Result = Align(matrix, BASE)
        Result.to_csv(path + "行业和国家因子\\" + factor_name + ".csv")
    return None


def ProcessStyleFactorSize(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Size".
    '''
    # Preprocess the descriptive variable LNCAP.
    LNCAP = pd.read_csv(path + "描述变量原始值\\" + "Size_LNCAP.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    LNCAP = process_factor(LNCAP, weight_df, I, mad=True)

    LNCAP_matrix = pd.pivot(LNCAP, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    LNCAP_matrix.to_csv(path + "描述变量\\" + "Size_LNCAP.csv")

    # Weight to obtain the style factor "Size"
    Size_matrix = LNCAP_matrix
    Size_matrix.to_csv(path + "风格因子\\" + "Size.csv")
    return None


def ProcessStyleFactorMidCapitalization(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Mid Capitalization".
    '''
    # Preprocess the descriptive variable MIDCAP.
    MIDCAP = pd.read_csv(path + "描述变量原始值\\" + "Mid Capitalization_MIDCAP.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    MIDCAP = process_factor(MIDCAP, weight_df, I, mad=True)

    MIDCAP_matrix = pd.pivot(MIDCAP, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    MIDCAP_matrix.to_csv(path + "描述变量\\" + "Mid Capitalization_MIDCAP.csv")

    # Weight to obtain the style factor "Mid Capitalization"
    Mid_Capitalization_matrix = MIDCAP_matrix
    Mid_Capitalization_matrix.to_csv(path + "风格因子\\" + "Mid Capitalization.csv")
    return None


def ProcessStyleFactorBeta(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Beta".
    '''
    # Preprocess the descriptive variable HBETA.
    HBETA = pd.read_csv(path + "描述变量原始值\\" + "Beta_HBETA.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    HBETA = process_factor(HBETA, weight_df, I, mad=True)

    HBETA_matrix = pd.pivot(HBETA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    HBETA_matrix.to_csv(path + "描述变量\\" + "Beta_HBETA.csv")

    # Weight to obtain the style factor "Beta"
    Beta_matrix = HBETA_matrix
    Beta_matrix.to_csv(path + "风格因子\\" + "Beta.csv")
    return None


def ProcessStyleFactorResidualVolatility(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Residual Volatility".
    '''
    # Preprocess the descriptive variables HSIGMA, DASTD, and CMRA.
    HSIGMA = pd.read_csv(path + "描述变量原始值\\" + "Residual Volatility_HSIGMA.csv", index_col=0)
    DASTD = pd.read_csv(path + "描述变量原始值\\" + "Residual Volatility_DASTD.csv", index_col=0)
    CMRA = pd.read_csv(path + "描述变量原始值\\" + "Residual Volatility_CMRA.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    HSIGMA = process_factor(HSIGMA, weight_df, I, mad=True)
    DASTD = process_factor(DASTD, weight_df, I, mad=True)
    CMRA = process_factor(CMRA, weight_df, I, mad=True)

    HSIGMA_matrix = pd.pivot(HSIGMA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    HSIGMA_matrix.to_csv(path + "描述变量\\" + "Residual Volatility_HSIGMA.csv")

    DASTD_matrix = pd.pivot(DASTD, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    DASTD_matrix.to_csv(path + "描述变量\\" + "Residual Volatility_DASTD.csv")

    CMRA_matrix = pd.pivot(CMRA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    CMRA_matrix.to_csv(path + "描述变量\\" + "Residual Volatility_CMRA.csv")

    # Weight to obtain the style factor "Residual Volatility"
    Residual_Volatility = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Residual_Volatility['FACTOR'] = (HSIGMA['FACTOR'] + DASTD['FACTOR'] + CMRA['FACTOR']) / 3
    # Standardize again after weighting.
    Residual_Volatility = standardize_factor(Residual_Volatility)
    Residual_Volatility_matrix = pd.pivot(Residual_Volatility, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Residual_Volatility_matrix.to_csv(path + "风格因子\\" + "Residual Volatility.csv")
    return None


def ProcessStyleFactorLiquidity(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Liquidity".
    '''
    # Preprocess the descriptive variables ATVR, STOA, STOM, and STOQ.
    ATVR = pd.read_csv(path + "描述变量原始值\\" + "Liquidity_ATVR.csv", index_col=0)
    STOA = pd.read_csv(path + "描述变量原始值\\" + "Liquidity_STOA.csv", index_col=0)
    STOM = pd.read_csv(path + "描述变量原始值\\" + "Liquidity_STOM.csv", index_col=0)
    STOQ = pd.read_csv(path + "描述变量原始值\\" + "Liquidity_STOQ.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    ATVR = process_factor(ATVR, weight_df, I, mad=True)
    STOA = process_factor(STOA, weight_df, I, mad=True)
    STOM = process_factor(STOM, weight_df, I, mad=True)
    STOQ = process_factor(STOQ, weight_df, I, mad=True)

    ATVR_matrix = pd.pivot(ATVR, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ATVR_matrix.to_csv(path + "描述变量\\" + "Liquidity_ATVR.csv")

    STOA_matrix = pd.pivot(STOA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    STOA_matrix.to_csv(path + "描述变量\\" + "Liquidity_STOA.csv")

    STOM_matrix = pd.pivot(STOM, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    STOM_matrix.to_csv(path + "描述变量\\" + "Liquidity_STOM.csv")

    STOQ_matrix = pd.pivot(STOQ, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    STOQ_matrix.to_csv(path + "描述变量\\" + "Liquidity_STOQ.csv")

    # Weight to obtain the style factor "Liquidity"
    Liquidity = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Liquidity['FACTOR'] = (ATVR['FACTOR'] + STOA['FACTOR'] + STOM['FACTOR'] + STOQ['FACTOR']) / 4
    # Standardize again after weighting.
    Liquidity = standardize_factor(Liquidity)
    Liquidity_matrix = pd.pivot(Liquidity, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Liquidity_matrix.to_csv(path + "风格因子\\" + "Liquidity.csv")
    return None


def ProcessStyleFactorShortTermReversal(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Short-Term Reversal".
    '''
    # Preprocess the descriptive variable STREV.
    STREV = pd.read_csv(path + "描述变量原始值\\" + "Short-Term Reversal_STREV.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    STREV = process_factor(STREV, weight_df, I, mad=True)

    STREV_matrix = pd.pivot(STREV, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    STREV_matrix.to_csv(path + "描述变量\\" + "Short-Term Reversal_STREV.csv")

    # Weight to obtain the style factor "Short-Term Reversal"
    Short_Term_Reversal_matrix = STREV_matrix
    Short_Term_Reversal_matrix.to_csv(path + "风格因子\\" + "Short-Term Reversal.csv")
    return None


def ProcessStyleFactorSeasonality(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Seasonality".
    '''
    # Preprocess the descriptive variable SEASON.
    SEASON = pd.read_csv(path + "描述变量原始值\\" + "Seasonality_SEASON.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    SEASON = process_factor(SEASON, weight_df, I, mad=True)

    SEASON_matrix = pd.pivot(SEASON, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    SEASON_matrix.to_csv(path + "描述变量\\" + "Seasonality_SEASON.csv")

    # Weight to obtain the style factor "Seasonality"
    Seasonality_matrix = SEASON_matrix
    Seasonality_matrix.to_csv(path + "风格因子\\" + "Seasonality.csv")
    return None


def ProcessStyleFactorIndustryMomentum(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Industry Momentum".
    '''
    # Preprocess the descriptive variable INDMOM.
    INDMOM = pd.read_csv(path + "描述变量原始值\\" + "Industry Momentum_INDMOM.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    INDMOM = process_factor(INDMOM, weight_df, I, mad=True)

    INDMOM_matrix = pd.pivot(INDMOM, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    INDMOM_matrix.to_csv(path + "描述变量\\" + "Industry Momentum_INDMOM.csv")

    # Weight to obtain the style factor "Industry Momentum"
    Industry_Momentum_matrix = INDMOM_matrix
    Industry_Momentum_matrix.to_csv(path + "风格因子\\" + "Industry Momentum.csv")
    return None


def ProcessStyleFactorMomentum(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Momentum".
    '''
    # Preprocess the descriptive variables RSTR and HALPHA.
    RSTR = pd.read_csv(path + "描述变量原始值\\" + "Momentum_RSTR.csv", index_col=0)
    HALPHA = pd.read_csv(path + "描述变量原始值\\" + "Momentum_HALPHA.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    RSTR = process_factor(RSTR, weight_df, I, mad=True)
    HALPHA = process_factor(HALPHA, weight_df, I, mad=True)

    RSTR_matrix = pd.pivot(RSTR, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    RSTR_matrix.to_csv(path + "描述变量\\" + "Momentum_RSTR.csv")

    HALPHA_matrix = pd.pivot(HALPHA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    HALPHA_matrix.to_csv(path + "描述变量\\" + "Momentum_HALPHA.csv")

    # Weight to obtain the style factor "Momentum"
    Momentum = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Momentum['FACTOR'] = (RSTR['FACTOR'] + HALPHA['FACTOR']) / 2
    # Standardize again after weighting.
    Momentum = standardize_factor(Momentum)
    Momentum_matrix = pd.pivot(Momentum, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Momentum_matrix.to_csv(path + "风格因子\\" + "Momentum.csv")
    return None


def ProcessStyleFactorLeverage(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Leverage".
    '''
    # Preprocess the descriptive variables MLEV, BLEV, and DTOA.
    MLEV = pd.read_csv(path + "描述变量原始值\\" + "Leverage_MLEV.csv", index_col=0)
    BLEV = pd.read_csv(path + "描述变量原始值\\" + "Leverage_BLEV.csv", index_col=0)
    DTOA = pd.read_csv(path + "描述变量原始值\\" + "Leverage_DTOA.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    MLEV = process_factor(MLEV, weight_df, I, mad=True)
    BLEV = process_factor(BLEV, weight_df, I, mad=True)
    DTOA = process_factor(DTOA, weight_df, I, mad=True)

    MLEV_matrix = pd.pivot(MLEV, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    MLEV_matrix.to_csv(path + "描述变量\\" + "Leverage_MLEV.csv")

    BLEV_matrix = pd.pivot(BLEV, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    BLEV_matrix.to_csv(path + "描述变量\\" + "Leverage_BLEV.csv")

    DTOA_matrix = pd.pivot(DTOA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    DTOA_matrix.to_csv(path + "描述变量\\" + "Leverage_DTOA.csv")

    # Weight to obtain the style factor "Leverage"
    Leverage = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Leverage['FACTOR'] = (MLEV['FACTOR'] + BLEV['FACTOR'] + DTOA['FACTOR']) / 3
    # Standardize again after weighting.
    Leverage = standardize_factor(Leverage)
    Leverage_matrix = pd.pivot(Leverage, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Leverage_matrix.to_csv(path + "风格因子\\" + "Leverage.csv")
    return None


def ProcessStyleFactorEarningsVariability(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Earnings Variability".
    '''
    # Preprocess the descriptive variables VSAL, VERN, VFLO, and ETOPF_STD.
    VSAL = pd.read_csv(path + "描述变量原始值\\" + "Earnings Variability_VSAL.csv", index_col=0)
    VERN = pd.read_csv(path + "描述变量原始值\\" + "Earnings Variability_VERN.csv", index_col=0)
    VFLO = pd.read_csv(path + "描述变量原始值\\" + "Earnings Variability_VFLO.csv", index_col=0)
    ETOPF_STD = pd.read_csv(path + "描述变量原始值\\" + "Earnings Variability_ETOPF_STD.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    VSAL = process_factor(VSAL, weight_df, I, mad=True)
    VERN = process_factor(VERN, weight_df, I, mad=True)
    VFLO = process_factor(VFLO, weight_df, I, mad=True)
    ETOPF_STD = process_factor(ETOPF_STD, weight_df, I, mad=True)

    VSAL_matrix = pd.pivot(VSAL, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VSAL_matrix.to_csv(path + "描述变量\\" + "Earnings Variability_VSAL.csv")

    VERN_matrix = pd.pivot(VERN, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VERN_matrix.to_csv(path + "描述变量\\" + "Earnings Variability_VERN.csv")

    VFLO_matrix = pd.pivot(VFLO, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VFLO_matrix.to_csv(path + "描述变量\\" + "Earnings Variability_VFLO.csv")

    ETOPF_STD_matrix = pd.pivot(ETOPF_STD, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ETOPF_STD_matrix.to_csv(path + "描述变量\\" + "Earnings Variability_ETOPF_STD.csv")

    # Weight to obtain the style factor "Earnings Variability"
    Earnings_Variability = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Earnings_Variability['FACTOR'] = (VSAL['FACTOR'] + VERN['FACTOR'] + VFLO['FACTOR'] + ETOPF_STD['FACTOR']) / 4
    # Standardize again after weighting.
    Earnings_Variability = standardize_factor(Earnings_Variability)
    Earnings_Variability_matrix = pd.pivot(Earnings_Variability, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Earnings_Variability_matrix.to_csv(path + "风格因子\\" + "Earnings Variability.csv")
    return None


def ProcessStyleFactorEarningsQuality(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Earnings Quality".
    '''
    # Preprocess the descriptive variables ABS and ACF.
    ABS = pd.read_csv(path + "描述变量原始值\\" + "Earnings Quality_ABS.csv", index_col=0)
    ACF = pd.read_csv(path + "描述变量原始值\\" + "Earnings Quality_ACF.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    ABS = process_factor(ABS, weight_df, I, mad=True)
    ACF = process_factor(ACF, weight_df, I, mad=True)

    ABS_matrix = pd.pivot(ABS, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ABS_matrix.to_csv(path + "描述变量\\" + "Earnings Quality_ABS.csv")

    ACF_matrix = pd.pivot(ACF, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ACF_matrix.to_csv(path + "描述变量\\" + "Earnings Quality_ACF.csv")

    # Weight to obtain the style factor "Earnings Quality"
    Earnings_Quality = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Earnings_Quality['FACTOR'] = (ABS['FACTOR'] + ACF['FACTOR']) / 2
    # Standardize again after weighting.
    Earnings_Quality = standardize_factor(Earnings_Quality)
    Earnings_Quality_matrix = pd.pivot(Earnings_Quality, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Earnings_Quality_matrix.to_csv(path + "风格因子\\" + "Earnings Quality.csv")
    return None


def ProcessStyleFactorProfitability(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Profitability".
    '''
    # Preprocess the descriptive variables ATO, GP, GPM, and ROA.
    ATO = pd.read_csv(path + "描述变量原始值\\" + "Profitability_ATO.csv", index_col=0)
    GP = pd.read_csv(path + "描述变量原始值\\" + "Profitability_GP.csv", index_col=0)
    GPM = pd.read_csv(path + "描述变量原始值\\" + "Profitability_GPM.csv", index_col=0)
    ROA = pd.read_csv(path + "描述变量原始值\\" + "Profitability_ROA.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    ATO = process_factor(ATO, weight_df, I, mad=True)
    GP = process_factor(GP, weight_df, I, mad=True)
    GPM = process_factor(GPM, weight_df, I, mad=True)
    ROA = process_factor(ROA, weight_df, I, mad=True)

    ATO_matrix = pd.pivot(ATO, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ATO_matrix.to_csv(path + "描述变量\\" + "Profitability_ATO.csv")

    GP_matrix = pd.pivot(GP, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    GP_matrix.to_csv(path + "描述变量\\" + "Profitability_GP.csv")

    GPM_matrix = pd.pivot(GPM, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    GPM_matrix.to_csv(path + "描述变量\\" + "Profitability_GPM.csv")

    ROA_matrix = pd.pivot(ROA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ROA_matrix.to_csv(path + "描述变量\\" + "Profitability_ROA.csv")

    # Weight to obtain the style factor "Profitability"
    Profitability = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Profitability['FACTOR'] = (ATO['FACTOR'] + GP['FACTOR'] + GPM['FACTOR'] + ROA['FACTOR']) / 4
    # Standardize again after weighting.
    Profitability = standardize_factor(Profitability)
    Profitability_matrix = pd.pivot(Profitability, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Profitability_matrix.to_csv(path + "风格因子\\" + "Profitability.csv")
    return None


def ProcessStyleFactorInvestmentQuality(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Investment Quality".
    '''
    # Preprocess the descriptive variables AGRO, IGRO, and CXGRO.
    AGRO = pd.read_csv(path + "描述变量原始值\\" + "Investment Quality_AGRO.csv", index_col=0)
    IGRO = pd.read_csv(path + "描述变量原始值\\" + "Investment Quality_IGRO.csv", index_col=0)
    CXGRO = pd.read_csv(path + "描述变量原始值\\" + "Investment Quality_CXGRO.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    AGRO = process_factor(AGRO, weight_df, I, mad=True)
    IGRO = process_factor(IGRO, weight_df, I, mad=False)  # For IGRO, no outlier removal is performed due to large differences across stocks.
    CXGRO = process_factor(CXGRO, weight_df, I, mad=True)

    AGRO_matrix = pd.pivot(AGRO, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    AGRO_matrix.to_csv(path + "描述变量\\" + "Investment Quality_AGRO.csv")

    IGRO_matrix = pd.pivot(IGRO, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    IGRO_matrix.to_csv(path + "描述变量\\" + "Investment Quality_IGRO.csv")

    CXGRO_matrix = pd.pivot(CXGRO, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    CXGRO_matrix.to_csv(path + "描述变量\\" + "Investment Quality_CXGRO.csv")

    # Weight to obtain the style factor "Investment Quality"
    Investment_Quality = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Investment_Quality['FACTOR'] = (AGRO['FACTOR'] + IGRO['FACTOR'] + CXGRO['FACTOR']) / 3
    # Standardize again after weighting.
    Investment_Quality = standardize_factor(Investment_Quality)
    Investment_Quality_matrix = pd.pivot(Investment_Quality, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Investment_Quality_matrix.to_csv(path + "风格因子\\" + "Investment Quality.csv")
    return None


def ProcessStyleFactorBooktoPrice(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Book-to-Price".
    '''
    # Preprocess the descriptive variable BTOP.
    BTOP = pd.read_csv(path + "描述变量原始值\\" + "Book-to-Price_BTOP.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    BTOP = process_factor(BTOP, weight_df, I, mad=True)

    BTOP_matrix = pd.pivot(BTOP, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    BTOP_matrix.to_csv(path + "描述变量\\" + "Book-to-Price_BTOP.csv")

    # Weight to obtain the style factor "Book-to-Price"
    Book_to_Price_matrix = BTOP_matrix
    Book_to_Price_matrix.to_csv(path + "风格因子\\" + "Book-to-Price.csv")
    return None


def ProcessStyleFactorEarningsYield(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Earnings Yield".
    '''
    # Preprocess the descriptive variables ETOPF, CETOP, ETOP, and EM.
    ETOPF = pd.read_csv(path + "描述变量原始值\\" + "Earnings Yield_ETOPF.csv", index_col=0)
    CETOP = pd.read_csv(path + "描述变量原始值\\" + "Earnings Yield_CETOP.csv", index_col=0)
    ETOP = pd.read_csv(path + "描述变量原始值\\" + "Earnings Yield_ETOP.csv", index_col=0)
    EM = pd.read_csv(path + "描述变量原始值\\" + "Earnings Yield_EM.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    ETOPF = process_factor(ETOPF, weight_df, I, mad=True)
    CETOP = process_factor(CETOP, weight_df, I, mad=True)
    ETOP = process_factor(ETOP, weight_df, I, mad=True)
    EM = process_factor(EM, weight_df, I, mad=True)

    ETOPF_matrix = pd.pivot(ETOPF, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ETOPF_matrix.to_csv(path + "描述变量\\" + "Earnings Yield_ETOPF.csv")

    CETOP_matrix = pd.pivot(CETOP, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    CETOP_matrix.to_csv(path + "描述变量\\" + "Earnings Yield_CETOP.csv")

    ETOP_matrix = pd.pivot(ETOP, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ETOP_matrix.to_csv(path + "描述变量\\" + "Earnings Yield_ETOP.csv")

    EM_matrix = pd.pivot(EM, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    EM_matrix.to_csv(path + "描述变量\\" + "Earnings Yield_EM.csv")

    # Weight to obtain the style factor "Earnings Yield"
    Earnings_Yield = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Earnings_Yield['FACTOR'] = (ETOPF['FACTOR'] + CETOP['FACTOR'] + ETOP['FACTOR'] + EM['FACTOR']) / 4
    # Standardize again after weighting.
    Earnings_Yield = standardize_factor(Earnings_Yield)
    Earnings_Yield_matrix = pd.pivot(Earnings_Yield, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Earnings_Yield_matrix.to_csv(path + "风格因子\\" + "Earnings Yield.csv")
    return None


def ProcessStyleFactorLongTermReversal(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Long-Term Reversal".
    '''
    # Preprocess the descriptive variables LTHALPHA and LTRSTR.
    LTHALPHA = pd.read_csv(path + "描述变量原始值\\" + "Long-Term Reversal_LTHALPHA.csv", index_col=0)
    LTRSTR = pd.read_csv(path + "描述变量原始值\\" + "Long-Term Reversal_LTRSTR.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    LTHALPHA = process_factor(LTHALPHA, weight_df, I, mad=True)
    LTRSTR = process_factor(LTRSTR, weight_df, I, mad=True)

    LTHALPHA_matrix = pd.pivot(LTHALPHA, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    LTHALPHA_matrix.to_csv(path + "描述变量\\" + "Long-Term Reversal_LTHALPHA.csv")

    LTRSTR_matrix = pd.pivot(LTRSTR, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    LTRSTR_matrix.to_csv(path + "描述变量\\" + "Long-Term Reversal_LTRSTR.csv")

    # Weight to obtain the style factor "Long-Term Reversal"
    Long_Term_Reversal = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Long_Term_Reversal['FACTOR'] = (LTHALPHA['FACTOR'] + LTRSTR['FACTOR']) / 2
    # Standardize again after weighting.
    Long_Term_Reversal = standardize_factor(Long_Term_Reversal)

    Long_Term_Reversal_matrix = pd.pivot(Long_Term_Reversal, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Long_Term_Reversal_matrix.to_csv(path + "风格因子\\" + "Long-Term Reversal.csv")
    return None


def ProcessStyleFactorGrowth(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Growth".
    '''
    # Preprocess the descriptive variables EGRO, EGRLF, and SGRO.
    EGRO = pd.read_csv(path + "描述变量原始值\\" + "Growth_EGRO.csv", index_col=0)
    EGRLF = pd.read_csv(path + "描述变量原始值\\" + "Growth_EGRLF.csv", index_col=0)
    SGRO = pd.read_csv(path + "描述变量原始值\\" + "Growth_SGRO.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    EGRO = process_factor(EGRO, weight_df, I, mad=True)
    EGRLF = process_factor(EGRLF, weight_df, I, mad=True)
    SGRO = process_factor(SGRO, weight_df, I, mad=True)

    EGRO_matrix = pd.pivot(EGRO, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    EGRO_matrix.to_csv(path + "描述变量\\" + "Growth_EGRO.csv")

    EGRLF_matrix = pd.pivot(EGRLF, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    EGRLF_matrix.to_csv(path + "描述变量\\" + "Growth_EGRLF.csv")

    SGRO_matrix = pd.pivot(SGRO, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    SGRO_matrix.to_csv(path + "描述变量\\" + "Growth_SGRO.csv")

    # Weight to obtain the style factor "Growth"
    Growth = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Growth['FACTOR'] = (EGRO['FACTOR'] + EGRLF['FACTOR'] + SGRO['FACTOR']) / 3
    # Standardize again after weighting.
    Growth = standardize_factor(Growth)

    Growth_matrix = pd.pivot(Growth, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Growth_matrix.to_csv(path + "风格因子\\" + "Growth.csv")
    return None


def ProcessStyleFactorAnalystSentiment(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Analyst Sentiment".
    '''
    # Preprocess the descriptive variables ETOPF_C, EPSF_C, and RR.
    ETOPF_C = pd.read_csv(path + "描述变量原始值\\" + "Analyst Sentiment_ETOPF_C.csv", index_col=0)
    EPSF_C = pd.read_csv(path + "描述变量原始值\\" + "Analyst Sentiment_EPSF_C.csv", index_col=0)
    RR = pd.read_csv(path + "描述变量原始值\\" + "Analyst Sentiment_RR.csv", index_col=0)
    # Factor processing (outlier removal, standardization, missing value imputation)
    ETOPF_C = process_factor(ETOPF_C, weight_df, I, mad=True)
    EPSF_C = process_factor(EPSF_C, weight_df, I, mad=True)
    RR = process_factor(RR, weight_df, I, mad=True)

    ETOPF_C_matrix = pd.pivot(ETOPF_C, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    ETOPF_C_matrix.to_csv(path + "描述变量\\" + "Analyst Sentiment_ETOPF_C.csv")

    EPSF_C_matrix = pd.pivot(EPSF_C, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    EPSF_C_matrix.to_csv(path + "描述变量\\" + "Analyst Sentiment_EPSF_C.csv")

    RR_matrix = pd.pivot(RR, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    RR_matrix.to_csv(path + "描述变量\\" + "Analyst Sentiment_RR.csv")

    # Weight to obtain the style factor "Analyst Sentiment"
    Analyst_Sentiment = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Analyst_Sentiment['FACTOR'] = (ETOPF_C['FACTOR'] + EPSF_C['FACTOR'] + RR['FACTOR']) / 3
    # Standardize again after weighting.
    Analyst_Sentiment = standardize_factor(Analyst_Sentiment)
    Analyst_Sentiment_matrix = pd.pivot(Analyst_Sentiment, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Analyst_Sentiment_matrix.to_csv(path + "风格因子\\" + "Analyst Sentiment.csv")
    return None


def ProcessStyleFactorDividendYield(weight_df, I, path):
    '''
    Process and weight to obtain the style factor "Dividend Yield".
    '''
    # Preprocess the descriptive variables DTOPF and DTOP.
    DTOPF = pd.read_csv(path + "描述变量原始值\\" + "Dividend Yield_DTOPF.csv", index_col=0)
    DTOP = pd.read_csv(path + "描述变量原始值\\" + "Dividend Yield_DTOP.csv", index_col=0)
    # Factor processing (no outlier removal for these factors due to large differences across stocks)
    DTOPF = process_factor(DTOPF, weight_df, I, mad=False)
    DTOP = process_factor(DTOP, weight_df, I, mad=False)

    DTOPF_matrix = pd.pivot(DTOPF, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    DTOPF_matrix.to_csv(path + "描述变量\\" + "Dividend Yield_DTOPF.csv")

    DTOP_matrix = pd.pivot(DTOP, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    DTOP_matrix.to_csv(path + "描述变量\\" + "Dividend Yield_DTOP.csv")

    # Weight to obtain the style factor "Dividend Yield"
    Dividend_Yield = weight_df[['S_INFO_WINDCODE', 'TRADE_DT', 'WEIGHT']]
    Dividend_Yield['FACTOR'] = (DTOPF['FACTOR'] + DTOP['FACTOR']) / 2
    # Standardize again after weighting.
    Dividend_Yield = standardize_factor(Dividend_Yield)
    Dividend_Yield_matrix = pd.pivot(Dividend_Yield, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    Dividend_Yield_matrix.to_csv(path + "风格因子\\" + "Dividend Yield.csv")
    return None


def ProcessStyleFactor(weight_df, I, path):
    '''
    Process and weight to obtain all style factors.
    '''
    if not os.path.exists(path + "描述变量\\"):
        os.makedirs(path + "描述变量\\")
    if not os.path.exists(path + "风格因子\\"):
        os.makedirs(path + "风格因子\\")
    # Process and weight the style factor "Size"
    print('Processing and weighting style factor Size')
    ProcessStyleFactorSize(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Mid Capitalization"
    print('Processing and weighting style factor Mid Capitalization')
    ProcessStyleFactorMidCapitalization(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Beta"
    print('Processing and weighting style factor Beta')
    ProcessStyleFactorBeta(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Residual Volatility"
    print('Processing and weighting style factor Residual Volatility')
    ProcessStyleFactorResidualVolatility(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Liquidity"
    print('Processing and weighting style factor Liquidity')
    ProcessStyleFactorLiquidity(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Short-Term Reversal"
    print('Processing and weighting style factor Short-Term Reversal')
    ProcessStyleFactorShortTermReversal(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Seasonality"
    print('Processing and weighting style factor Seasonality')
    ProcessStyleFactorSeasonality(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Industry Momentum"
    print('Processing and weighting style factor Industry Momentum')
    ProcessStyleFactorIndustryMomentum(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Momentum"
    print('Processing and weighting style factor Momentum')
    ProcessStyleFactorMomentum(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Leverage"
    print('Processing and weighting style factor Leverage')
    ProcessStyleFactorLeverage(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Earnings Variability"
    print('Processing and weighting style factor Earnings Variability')
    ProcessStyleFactorEarningsVariability(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Earnings Quality"
    print('Processing and weighting style factor Earnings Quality')
    ProcessStyleFactorEarningsQuality(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Profitability"
    print('Processing and weighting style factor Profitability')
    ProcessStyleFactorProfitability(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Investment Quality"
    print('Processing and weighting style factor Investment Quality')
    ProcessStyleFactorInvestmentQuality(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Book-to-Price"
    print('Processing and weighting style factor Book-to-Price')
    ProcessStyleFactorBooktoPrice(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Earnings Yield"
    print('Processing and weighting style factor Earnings Yield')
    ProcessStyleFactorEarningsYield(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Long-Term Reversal"
    print('Processing and weighting style factor Long-Term Reversal')
    ProcessStyleFactorLongTermReversal(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Growth"
    print('Processing and weighting style factor Growth')
    ProcessStyleFactorGrowth(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Analyst Sentiment"
    print('Processing and weighting style factor Analyst Sentiment')
    ProcessStyleFactorAnalystSentiment(weight_df=weight_df, I=I, path=path)
    # Process and weight the style factor "Dividend Yield"
    print('Processing and weighting style factor Dividend Yield')
    ProcessStyleFactorDividendYield(weight_df=weight_df, I=I, path=path)
    return None


def Orthogonalization_Liquidity(factor):
    '''
    Orthogonalize Liquidity with respect to Size.
    '''
    factor['Liquidity_new'] = np.nan

    def cal_Liquidity_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20050912':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Size']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Liquidity']
            # Fill missing values with the market median if needed
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Liquidity_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Liquidity")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Liquidity_new).reset_index(drop=True)
    factor = factor.drop('Liquidity', axis=1)
    factor.rename(columns={'Liquidity_new': 'Liquidity'}, inplace=True)
    return factor


def Orthogonalization_ResidualVolatility(factor):
    '''
    Orthogonalize Residual Volatility with respect to Beta and Size.
    '''
    factor['Residual Volatility_new'] = np.nan

    def cal_Residual_Volatility_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20070206':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, ['Beta', 'Size']]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Residual Volatility']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Residual Volatility_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Residual Volatility")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Residual_Volatility_new).reset_index(drop=True)
    factor = factor.drop('Residual Volatility', axis=1)
    factor.rename(columns={'Residual Volatility_new': 'Residual Volatility'}, inplace=True)
    return factor


def Orthogonalization_LongTermReversal(factor):
    '''
    Orthogonalize Long-Term Reversal with respect to Momentum.
    '''
    factor['Long-Term Reversal_new'] = np.nan

    def cal_Long_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Momentum']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Long-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Long-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Long-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Long_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Long-Term Reversal', axis=1)
    factor.rename(columns={'Long-Term Reversal_new': 'Long-Term Reversal'}, inplace=True)
    return factor


def Orthogonalization_AnalystSentiment(factor):
    '''
    Orthogonalize the short-term factor Analyst Sentiment with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Analyst Sentiment_new'] = np.nan

    def cal_Analyst_Sentiment_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Analyst Sentiment']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Analyst Sentiment_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Analyst Sentiment")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Analyst_Sentiment_new).reset_index(drop=True)
    factor = factor.drop('Analyst Sentiment', axis=1)
    factor.rename(columns={'Analyst Sentiment_new': 'Analyst Sentiment'}, inplace=True)
    return factor


def Orthogonalization_IndustryMomentum(factor):
    '''
    Orthogonalize the short-term factor Industry Momentum with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Industry Momentum_new'] = np.nan

    def cal_Industry_Momentum_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Industry Momentum']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Industry Momentum_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Industry Momentum")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Industry_Momentum_new).reset_index(drop=True)
    factor = factor.drop('Industry Momentum', axis=1)
    factor.rename(columns={'Industry Momentum_new': 'Industry Momentum'}, inplace=True)
    return factor


def Orthogonalization_Seasonality(factor):
    '''
    Orthogonalize the short-term factor Seasonality with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Seasonality_new'] = np.nan

    def cal_Seasonality_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Seasonality']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Seasonality_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Seasonality")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Seasonality_new).reset_index(drop=True)
    factor = factor.drop('Seasonality', axis=1)
    factor.rename(columns={'Seasonality_new': 'Seasonality'}, inplace=True)
    return factor


def Orthogonalization_ShortTermReversal(factor):
    '''
    Orthogonalize the short-term factor Short-Term Reversal with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Short-Term Reversal_new'] = np.nan

    def cal_Short_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Short-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Short-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Short-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Short_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Short-Term Reversal', axis=1)
    factor.rename(columns={'Short-Term Reversal_new': 'Short-Term Reversal'}, inplace=True)
    return factor


def Orthogonalization(BASE, path):
    '''
    Orthogonalize the style factors.
    '''
    print('Orthogonalizing style factors')
    # Read style factors
    style_factor_names = ["Analyst Sentiment", "Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                          "Earnings Yield", "Growth", "Industry Momentum", "Investment Quality",
                          "Leverage", "Liquidity", "Long-Term Reversal", "Mid Capitalization", "Momentum",
                          "Profitability", "Residual Volatility", "Seasonality", "Short-Term Reversal", "Size"]
    factor = BASE
    for factor_name in tqdm(style_factor_names, desc="Reading style factors"):
        factor_tmp = pd.read_csv(path + "风格因子\\" + factor_name + ".csv", index_col=0)
        factor_tmp_df = factor_format_adjust(factor_tmp, factor_name)
        factor = pd.merge(factor, factor_tmp_df, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 1: Orthogonalize Residual Volatility with respect to Beta and Size.
    factor = Orthogonalization_ResidualVolatility(factor)

    # 2: Orthogonalize Long-Term Reversal with respect to Momentum.
    factor = Orthogonalization_LongTermReversal(factor)

    # 3: Orthogonalize Liquidity with respect to Size.
    factor = Orthogonalization_Liquidity(factor)

    # 4: Orthogonalize short-term factors (Analyst Sentiment, Industry Momentum, Seasonality, Short-Term Reversal) with respect to 16 long-term factors.
    factor = Orthogonalization_AnalystSentiment(factor)
    factor = Orthogonalization_IndustryMomentum(factor)
    factor = Orthogonalization_Seasonality(factor)
    factor = Orthogonalization_ShortTermReversal(factor)

    # Save orthogonalized style factors.
    for factor_name in tqdm(['Liquidity', 'Residual Volatility', 'Long-Term Reversal', 'Liquidity', 'Analyst Sentiment', 'Industry Momentum', 'Seasonality', 'Short-Term Reversal'], desc="Saving orthogonalized factors"):
        factor_tmp = factor[['S_INFO_WINDCODE', 'TRADE_DT', factor_name]]
        # Pivot to matrix form
        matrix = pd.pivot(factor_tmp, index='S_INFO_WINDCODE', columns='TRADE_DT', values=factor_name)
        matrix.to_csv(path + '风格因子\\' + factor_name + '.csv')
    return None


def ProcessFirstFactorSIZE(weight_df, path):
    '''
    Weight to obtain the first-level (CNLT) factor SIZE.
    '''
    Size = pd.read_csv(path + "风格因子\\" + "Size.csv", index_col=0)
    Mid_Capitalization = pd.read_csv(path + "风格因子\\" + "Mid Capitalization.csv", index_col=0)
    # Obtain the CNLT first-level factor SIZE by weighting: 0.9*Size - 0.1*Mid Capitalization.
    SIZE = 0.9 * Size - 0.1 * Mid_Capitalization

    SIZE = SIZE.stack().reset_index()
    SIZE = SIZE.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    SIZE = pd.merge(weight_df, SIZE, how='left')
    
    # Standardize after weighting.
    SIZE = standardize_factor(SIZE)
    SIZE_matrix = pd.pivot(SIZE, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    SIZE_matrix.to_csv(path + "CNLT大类因子\\" + "SIZE.csv")
    return None


def ProcessFirstFactorVOLATILITY(weight_df, path):
    '''
    Weight to obtain the first-level factor VOLATILITY.
    '''
    Beta = pd.read_csv(path + "风格因子\\" + "Beta.csv", index_col=0)
    Residual_Volatility = pd.read_csv(path + "风格因子\\" + "Residual Volatility.csv", index_col=0)
    # Obtain the CNLT first-level factor VOLATILITY: 0.6*Beta + 0.4*Residual Volatility.
    VOLATILITY = 0.6 * Beta + 0.4 * Residual_Volatility

    VOLATILITY = VOLATILITY.stack().reset_index()
    VOLATILITY = VOLATILITY.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    VOLATILITY = pd.merge(weight_df, VOLATILITY, how='left')
    
    # Standardize after weighting.
    VOLATILITY = standardize_factor(VOLATILITY)
    VOLATILITY_matrix = pd.pivot(VOLATILITY, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VOLATILITY_matrix.to_csv(path + "CNLT大类因子\\" + "VOLATILITY.csv")
    return None


def ProcessFirstFactorLIQUIDITY(weight_df, path):
    '''
    Weight to obtain the first-level factor LIQUIDITY.
    '''
    Liquidity = pd.read_csv(path + "风格因子\\" + "Liquidity.csv", index_col=0)
    # The CNLT first-level factor LIQUIDITY is obtained directly.
    LIQUIDITY = Liquidity
    LIQUIDITY.to_csv(path + "CNLT大类因子\\" + "LIQUIDITY.csv")
    return None


def ProcessFirstFactorMOMENTUM(weight_df, path):
    '''
    Weight to obtain the first-level factor MOMENTUM.
    '''
    Momentum = pd.read_csv(path + "风格因子\\" + "Momentum.csv", index_col=0)
    # The CNLT first-level factor MOMENTUM is obtained directly.
    MOMENTUM = Momentum
    MOMENTUM.to_csv(path + "CNLT大类因子\\" + "MOMENTUM.csv")
    return None


def ProcessFirstFactorQUALITY(weight_df, path):
    '''
    Weight to obtain the first-level factor QUALITY.
    '''
    Leverage = pd.read_csv(path + "风格因子\\" + "Leverage.csv", index_col=0)
    Earnings_Variability = pd.read_csv(path + "风格因子\\" + "Earnings Variability.csv", index_col=0)
    Earnings_Quality = pd.read_csv(path + "风格因子\\" + "Earnings Quality.csv", index_col=0)
    Profitability = pd.read_csv(path + "风格因子\\" + "Profitability.csv", index_col=0)
    Investment_Quality = pd.read_csv(path + "风格因子\\" + "Investment Quality.csv", index_col=0)
    # Obtain the CNLT first-level factor QUALITY:
    # QUALITY = -0.125*Leverage - 0.125*Earnings Variability + 0.25*Earnings Quality + 0.25*Profitability + 0.25*Investment Quality
    QUALITY = -0.125 * Leverage - 0.125 * Earnings_Variability + 0.25 * Earnings_Quality + 0.25 * Profitability + 0.25 * Investment_Quality

    QUALITY = QUALITY.stack().reset_index()
    QUALITY = QUALITY.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    QUALITY = pd.merge(weight_df, QUALITY, how='left')
    
    # Standardize after weighting.
    QUALITY = standardize_factor(QUALITY)
    QUALITY_matrix = pd.pivot(QUALITY, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    QUALITY_matrix.to_csv(path + "CNLT大类因子\\" + "QUALITY.csv")
    return None


def ProcessFirstFactorVALUE(weight_df, path):
    '''
    Weight to obtain the first-level factor VALUE.
    '''
    Book_to_Price = pd.read_csv(path + "风格因子\\" + "Book-to-Price.csv", index_col=0)
    Earnings_Yield = pd.read_csv(path + "风格因子\\" + "Earnings Yield.csv", index_col=0)
    Long_Term_Reversal = pd.read_csv(path + "风格因子\\" + "Long-Term Reversal.csv", index_col=0)

    # Obtain the CNLT first-level factor VALUE: 0.3*Book-to-Price + 0.6*Earnings Yield + 0.1*Long-Term Reversal
    VALUE = 0.3 * Book_to_Price + 0.6 * Earnings_Yield + 0.1 * Long_Term_Reversal

    VALUE = VALUE.stack().reset_index()
    VALUE = VALUE.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    VALUE = pd.merge(weight_df, VALUE, how='left')
    
    # Standardize after weighting.
    VALUE = standardize_factor(VALUE)
    VALUE_matrix = pd.pivot(VALUE, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VALUE_matrix.to_csv(path + "CNLT大类因子\\" + "VALUE.csv")
    return None


def ProcessFirstFactorGROWTH(weight_df, path):
    '''
    Weight to obtain the first-level factor GROWTH.
    '''
    Growth = pd.read_csv(path + "风格因子\\" + "Growth.csv", index_col=0)
    # The CNLT first-level factor GROWTH is obtained directly.
    GROWTH = Growth
    GROWTH.to_csv(path + "CNLT大类因子\\" + "GROWTH.csv")
    return None


def ProcessFirstFactorYIELD(weight_df, path):
    '''
    Weight to obtain the first-level factor YIELD.
    '''
    Dividend_Yield = pd.read_csv(path + "风格因子\\" + "Dividend Yield.csv", index_col=0)
    # The CNLT first-level factor YIELD is obtained directly.
    YIELD = Dividend_Yield
    YIELD.to_csv(path + "CNLT大类因子\\" + "YIELD.csv")
    return None


def ProcessFirstFactor(weight_df, path):
    '''
    Weight to obtain all first-level (CNLT) factors.
    '''
    if not os.path.exists(path + "CNLT大类因子\\"):
        os.makedirs(path + "CNLT大类因子\\")
    # Weight to obtain first-level factor SIZE.
    print('Weighting first-level factor SIZE')
    ProcessFirstFactorSIZE(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor VOLATILITY.
    print('Weighting first-level factor VOLATILITY')
    ProcessFirstFactorVOLATILITY(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor LIQUIDITY.
    print('Weighting first-level factor LIQUIDITY')
    ProcessFirstFactorLIQUIDITY(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor MOMENTUM.
    print('Weighting first-level factor MOMENTUM')
    ProcessFirstFactorMOMENTUM(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor QUALITY.
    print('Weighting first-level factor QUALITY')
    ProcessFirstFactorQUALITY(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor VALUE.
    print('Weighting first-level factor VALUE')
    ProcessFirstFactorVALUE(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor GROWTH.
    print('Weighting first-level factor GROWTH')
    ProcessFirstFactorGROWTH(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor YIELD.
    print('Weighting first-level factor YIELD')
    ProcessFirstFactorYIELD(weight_df=weight_df, path=path)

    return None


def Orthogonalization_Liquidity(factor):
    '''
    Orthogonalize Liquidity with respect to Size.
    '''
    factor['Liquidity_new'] = np.nan

    def cal_Liquidity_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20050912':  # The latest starting date for this factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Size']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Liquidity']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Liquidity_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Liquidity")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Liquidity_new).reset_index(drop=True)
    factor = factor.drop('Liquidity', axis=1)
    factor.rename(columns={'Liquidity_new': 'Liquidity'}, inplace=True)
    return factor


def Orthogonalization_ResidualVolatility(factor):
    '''
    Orthogonalize Residual Volatility with respect to Beta and Size.
    '''
    factor['Residual Volatility_new'] = np.nan

    def cal_Residual_Volatility_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20070206':  # The latest starting date for this factor
            X = group[group['TRADE_DT'] == date].loc[:, ['Beta', 'Size']]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Residual Volatility']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Residual Volatility_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Residual Volatility")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Residual_Volatility_new).reset_index(drop=True)
    factor = factor.drop('Residual Volatility', axis=1)
    factor.rename(columns={'Residual Volatility_new': 'Residual Volatility'}, inplace=True)
    return factor


def Orthogonalization_LongTermReversal(factor):
    '''
    Orthogonalize Long-Term Reversal with respect to Momentum.
    '''
    factor['Long-Term Reversal_new'] = np.nan

    def cal_Long_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for this factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Momentum']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Long-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Long-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Long-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Long_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Long-Term Reversal', axis=1)
    factor.rename(columns={'Long-Term Reversal_new': 'Long-Term Reversal'}, inplace=True)
    return factor


def Orthogonalization_AnalystSentiment(factor):
    '''
    Orthogonalize the short-term factor Analyst Sentiment with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Analyst Sentiment_new'] = np.nan

    def cal_Analyst_Sentiment_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for this factor
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Analyst Sentiment']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Analyst Sentiment_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Analyst Sentiment")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Analyst_Sentiment_new).reset_index(drop=True)
    factor = factor.drop('Analyst Sentiment', axis=1)
    factor.rename(columns={'Analyst Sentiment_new': 'Analyst Sentiment'}, inplace=True)
    return factor


def Orthogonalization_IndustryMomentum(factor):
    '''
    Orthogonalize the short-term factor Industry Momentum with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Industry Momentum_new'] = np.nan

    def cal_Industry_Momentum_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Industry Momentum']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Industry Momentum_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Industry Momentum")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Industry_Momentum_new).reset_index(drop=True)
    factor = factor.drop('Industry Momentum', axis=1)
    factor.rename(columns={'Industry Momentum_new': 'Industry Momentum'}, inplace=True)
    return factor


def Orthogonalization_Seasonality(factor):
    '''
    Orthogonalize the short-term factor Seasonality with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Seasonality_new'] = np.nan

    def cal_Seasonality_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Seasonality']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Seasonality_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Seasonality")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Seasonality_new).reset_index(drop=True)
    factor = factor.drop('Seasonality', axis=1)
    factor.rename(columns={'Seasonality_new': 'Seasonality'}, inplace=True)
    return factor


def Orthogonalization_ShortTermReversal(factor):
    '''
    Orthogonalize the short-term factor Short-Term Reversal with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Short-Term Reversal_new'] = np.nan

    def cal_Short_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Short-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Short-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Short-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Short_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Short-Term Reversal', axis=1)
    factor.rename(columns={'Short-Term Reversal_new': 'Short-Term Reversal'}, inplace=True)
    return factor


def ProcessStyleFactorOrthogonalization(weight_df, I, path):
    '''
    Process and weight the style factors, then orthogonalize them.
    '''
    print('Processing and weighting style factors')
    # Read style factors
    style_factor_names = ["Analyst Sentiment", "Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                          "Earnings Yield", "Growth", "Industry Momentum", "Investment Quality",
                          "Leverage", "Liquidity", "Long-Term Reversal", "Mid Capitalization", "Momentum",
                          "Profitability", "Residual Volatility", "Seasonality", "Short-Term Reversal", "Size"]
    factor = BASE
    for factor_name in tqdm(style_factor_names, desc="Reading style factors"):
        factor_tmp = pd.read_csv(path + "风格因子\\" + factor_name + ".csv", index_col=0)
        factor_tmp_df = factor_format_adjust(factor_tmp, factor_name)
        factor = pd.merge(factor, factor_tmp_df, how='left', on=['S_INFO_WINDCODE', 'TRADE_DT'])

    # 1: Orthogonalize Residual Volatility with respect to Beta and Size.
    factor = Orthogonalization_ResidualVolatility(factor)
    # 2: Orthogonalize Long-Term Reversal with respect to Momentum.
    factor = Orthogonalization_LongTermReversal(factor)
    # 3: Orthogonalize Liquidity with respect to Size.
    factor = Orthogonalization_Liquidity(factor)
    # 4: Orthogonalize short-term factors (Analyst Sentiment, Industry Momentum, Seasonality, Short-Term Reversal) with respect to 16 long-term factors.
    factor = Orthogonalization_AnalystSentiment(factor)
    factor = Orthogonalization_IndustryMomentum(factor)
    factor = Orthogonalization_Seasonality(factor)
    factor = Orthogonalization_ShortTermReversal(factor)

    # Save the orthogonalized style factors.
    for factor_name in tqdm(['Liquidity', 'Residual Volatility', 'Long-Term Reversal', 'Liquidity', 'Analyst Sentiment',
                              'Industry Momentum', 'Seasonality', 'Short-Term Reversal'], desc="Saving orthogonalized factors"):
        factor_tmp = factor[['S_INFO_WINDCODE', 'TRADE_DT', factor_name]]
        # Pivot to matrix form.
        matrix = pd.pivot(factor_tmp, index='S_INFO_WINDCODE', columns='TRADE_DT', values=factor_name)
        matrix.to_csv(path + '风格因子\\' + factor_name + '.csv')
    return None


def ProcessFirstFactorSIZE(weight_df, path):
    '''
    Weight to obtain the first-level (CNLT) factor SIZE.
    '''
    Size = pd.read_csv(path + "风格因子\\" + "Size.csv", index_col=0)
    Mid_Capitalization = pd.read_csv(path + "风格因子\\" + "Mid Capitalization.csv", index_col=0)
    # Compute the CNLT first-level factor SIZE: 0.9 * Size - 0.1 * Mid Capitalization.
    SIZE = 0.9 * Size - 0.1 * Mid_Capitalization

    SIZE = SIZE.stack().reset_index()
    SIZE = SIZE.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    SIZE = pd.merge(weight_df, SIZE, how='left')

    # Standardize after weighting.
    SIZE = standardize_factor(SIZE)
    SIZE_matrix = pd.pivot(SIZE, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    SIZE_matrix.to_csv(path + "CNLT大类因子\\" + "SIZE.csv")
    return None


def ProcessFirstFactorVOLATILITY(weight_df, path):
    '''
    Weight to obtain the first-level factor VOLATILITY.
    '''
    Beta = pd.read_csv(path + "风格因子\\" + "Beta.csv", index_col=0)
    Residual_Volatility = pd.read_csv(path + "风格因子\\" + "Residual Volatility.csv", index_col=0)
    # Compute the CNLT first-level factor VOLATILITY: 0.6 * Beta + 0.4 * Residual Volatility.
    VOLATILITY = 0.6 * Beta + 0.4 * Residual_Volatility

    VOLATILITY = VOLATILITY.stack().reset_index()
    VOLATILITY = VOLATILITY.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    VOLATILITY = pd.merge(weight_df, VOLATILITY, how='left')

    # Standardize after weighting.
    VOLATILITY = standardize_factor(VOLATILITY)
    VOLATILITY_matrix = pd.pivot(VOLATILITY, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VOLATILITY_matrix.to_csv(path + "CNLT大类因子\\" + "VOLATILITY.csv")
    return None


def ProcessFirstFactorLIQUIDITY(weight_df, path):
    '''
    Weight to obtain the first-level factor LIQUIDITY.
    '''
    Liquidity = pd.read_csv(path + "风格因子\\" + "Liquidity.csv", index_col=0)
    LIQUIDITY = Liquidity
    LIQUIDITY.to_csv(path + "CNLT大类因子\\" + "LIQUIDITY.csv")
    return None


def ProcessFirstFactorMOMENTUM(weight_df, path):
    '''
    Weight to obtain the first-level factor MOMENTUM.
    '''
    Momentum = pd.read_csv(path + "风格因子\\" + "Momentum.csv", index_col=0)
    MOMENTUM = Momentum
    MOMENTUM.to_csv(path + "CNLT大类因子\\" + "MOMENTUM.csv")
    return None


def ProcessFirstFactorQUALITY(weight_df, path):
    '''
    Weight to obtain the first-level factor QUALITY.
    '''
    Leverage = pd.read_csv(path + "风格因子\\" + "Leverage.csv", index_col=0)
    Earnings_Variability = pd.read_csv(path + "风格因子\\" + "Earnings Variability.csv", index_col=0)
    Earnings_Quality = pd.read_csv(path + "风格因子\\" + "Earnings Quality.csv", index_col=0)
    Profitability = pd.read_csv(path + "风格因子\\" + "Profitability.csv", index_col=0)
    Investment_Quality = pd.read_csv(path + "风格因子\\" + "Investment Quality.csv", index_col=0)
    # Compute the CNLT first-level factor QUALITY:
    # QUALITY = -0.125 * Leverage - 0.125 * Earnings Variability + 0.25 * Earnings Quality + 0.25 * Profitability + 0.25 * Investment Quality
    QUALITY = -0.125 * Leverage - 0.125 * Earnings_Variability + 0.25 * Earnings_Quality + 0.25 * Profitability + 0.25 * Investment_Quality

    QUALITY = QUALITY.stack().reset_index()
    QUALITY = QUALITY.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    QUALITY = pd.merge(weight_df, QUALITY, how='left')

    # Standardize after weighting.
    QUALITY = standardize_factor(QUALITY)
    QUALITY_matrix = pd.pivot(QUALITY, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    QUALITY_matrix.to_csv(path + "CNLT大类因子\\" + "QUALITY.csv")
    return None


def ProcessFirstFactorVALUE(weight_df, path):
    '''
    Weight to obtain the first-level factor VALUE.
    '''
    Book_to_Price = pd.read_csv(path + "风格因子\\" + "Book-to-Price.csv", index_col=0)
    Earnings_Yield = pd.read_csv(path + "风格因子\\" + "Earnings Yield.csv", index_col=0)
    Long_Term_Reversal = pd.read_csv(path + "风格因子\\" + "Long-Term Reversal.csv", index_col=0)

    VALUE = 0.3 * Book_to_Price + 0.6 * Earnings_Yield + 0.1 * Long_Term_Reversal

    VALUE = VALUE.stack().reset_index()
    VALUE = VALUE.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    VALUE = pd.merge(weight_df, VALUE, how='left')

    # Standardize after weighting.
    VALUE = standardize_factor(VALUE)
    VALUE_matrix = pd.pivot(VALUE, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VALUE_matrix.to_csv(path + "CNLT大类因子\\" + "VALUE.csv")
    return None


def ProcessFirstFactorGROWTH(weight_df, path):
    '''
    Weight to obtain the first-level factor GROWTH.
    '''
    Growth = pd.read_csv(path + "风格因子\\" + "Growth.csv", index_col=0)
    GROWTH = Growth
    GROWTH.to_csv(path + "CNLT大类因子\\" + "GROWTH.csv")
    return None


def ProcessFirstFactorYIELD(weight_df, path):
    '''
    Weight to obtain the first-level factor YIELD.
    '''
    Dividend_Yield = pd.read_csv(path + "风格因子\\" + "Dividend Yield.csv", index_col=0)
    YIELD = Dividend_Yield
    YIELD.to_csv(path + "CNLT大类因子\\" + "YIELD.csv")
    return None


def ProcessFirstFactor(weight_df, path):
    '''
    Weight to obtain all first-level (CNLT) factors.
    '''
    if not os.path.exists(path + "CNLT大类因子\\"):
        os.makedirs(path + "CNLT大类因子\\")
    # Weight to obtain first-level factor SIZE.
    print('Weighting first-level factor SIZE')
    ProcessFirstFactorSIZE(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor VOLATILITY.
    print('Weighting first-level factor VOLATILITY')
    ProcessFirstFactorVOLATILITY(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor LIQUIDITY.
    print('Weighting first-level factor LIQUIDITY')
    ProcessFirstFactorLIQUIDITY(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor MOMENTUM.
    print('Weighting first-level factor MOMENTUM')
    ProcessFirstFactorMOMENTUM(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor QUALITY.
    print('Weighting first-level factor QUALITY')
    ProcessFirstFactorQUALITY(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor VALUE.
    print('Weighting first-level factor VALUE')
    ProcessFirstFactorVALUE(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor GROWTH.
    print('Weighting first-level factor GROWTH')
    ProcessFirstFactorGROWTH(weight_df=weight_df, path=path)
    # Weight to obtain first-level factor YIELD.
    print('Weighting first-level factor YIELD')
    ProcessFirstFactorYIELD(weight_df=weight_df, path=path)

    return None


def Orthogonalization_Liquidity(factor):
    '''
    Orthogonalize Liquidity with respect to Size.
    '''
    factor['Liquidity_new'] = np.nan

    def cal_Liquidity_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20050912':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Size']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Liquidity']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Liquidity_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Liquidity")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Liquidity_new).reset_index(drop=True)
    factor = factor.drop('Liquidity', axis=1)
    factor.rename(columns={'Liquidity_new': 'Liquidity'}, inplace=True)
    return factor


def Orthogonalization_ResidualVolatility(factor):
    '''
    Orthogonalize Residual Volatility with respect to Beta and Size.
    '''
    factor['Residual Volatility_new'] = np.nan

    def cal_Residual_Volatility_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20070206':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, ['Beta', 'Size']]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Residual Volatility']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Residual Volatility_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Residual Volatility")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Residual_Volatility_new).reset_index(drop=True)
    factor = factor.drop('Residual Volatility', axis=1)
    factor.rename(columns={'Residual Volatility_new': 'Residual Volatility'}, inplace=True)
    return factor


def Orthogonalization_LongTermReversal(factor):
    '''
    Orthogonalize Long-Term Reversal with respect to Momentum.
    '''
    factor['Long-Term Reversal_new'] = np.nan

    def cal_Long_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Momentum']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Long-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Long-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Long-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Long_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Long-Term Reversal', axis=1)
    factor.rename(columns={'Long-Term Reversal_new': 'Long-Term Reversal'}, inplace=True)
    return factor


def Orthogonalization_AnalystSentiment(factor):
    '''
    Orthogonalize the short-term factor Analyst Sentiment with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Analyst Sentiment_new'] = np.nan

    def cal_Analyst_Sentiment_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Analyst Sentiment']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Analyst Sentiment_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Analyst Sentiment")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Analyst_Sentiment_new).reset_index(drop=True)
    factor = factor.drop('Analyst Sentiment', axis=1)
    factor.rename(columns={'Analyst Sentiment_new': 'Analyst Sentiment'}, inplace=True)
    return factor


def Orthogonalization_IndustryMomentum(factor):
    '''
    Orthogonalize the short-term factor Industry Momentum with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Industry Momentum_new'] = np.nan

    def cal_Industry_Momentum_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Industry Momentum']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Industry Momentum_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Industry Momentum")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Industry_Momentum_new).reset_index(drop=True)
    factor = factor.drop('Industry Momentum', axis=1)
    factor.rename(columns={'Industry Momentum_new': 'Industry Momentum'}, inplace=True)
    return factor


def Orthogonalization_Seasonality(factor):
    '''
    Orthogonalize the short-term factor Seasonality with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Seasonality_new'] = np.nan

    def cal_Seasonality_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Seasonality']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Seasonality_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Seasonality")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Seasonality_new).reset_index(drop=True)
    factor = factor.drop('Seasonality', axis=1)
    factor.rename(columns={'Seasonality_new': 'Seasonality'}, inplace=True)
    return factor


def Orthogonalization_ShortTermReversal(factor):
    '''
    Orthogonalize the short-term factor Short-Term Reversal with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Short-Term Reversal_new'] = np.nan

    def cal_Short_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Short-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Short-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Short-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Short_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Short-Term Reversal', axis=1)
    factor.rename(columns={'Short-Term Reversal_new': 'Short-Term Reversal'}, inplace=True)
    return factor


def ProcessStyleFactor(weight_df, I, path):
    '''
    Process and weight to obtain all style factors.
    '''
    if not os.path.exists(path + "描述变量\\"):
        os.makedirs(path + "描述变量\\")
    if not os.path.exists(path + "风格因子\\"):
        os.makedirs(path + "风格因子\\")
    # Process and weight style factor Size
    print('Processing and weighting style factor Size')
    ProcessStyleFactorSize(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Mid Capitalization
    print('Processing and weighting style factor Mid Capitalization')
    ProcessStyleFactorMidCapitalization(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Beta
    print('Processing and weighting style factor Beta')
    ProcessStyleFactorBeta(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Residual Volatility
    print('Processing and weighting style factor Residual Volatility')
    ProcessStyleFactorResidualVolatility(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Liquidity
    print('Processing and weighting style factor Liquidity')
    ProcessStyleFactorLiquidity(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Short-Term Reversal
    print('Processing and weighting style factor Short-Term Reversal')
    ProcessStyleFactorShortTermReversal(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Seasonality
    print('Processing and weighting style factor Seasonality')
    ProcessStyleFactorSeasonality(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Industry Momentum
    print('Processing and weighting style factor Industry Momentum')
    ProcessStyleFactorIndustryMomentum(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Momentum
    print('Processing and weighting style factor Momentum')
    ProcessStyleFactorMomentum(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Leverage
    print('Processing and weighting style factor Leverage')
    ProcessStyleFactorLeverage(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Earnings Variability
    print('Processing and weighting style factor Earnings Variability')
    ProcessStyleFactorEarningsVariability(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Earnings Quality
    print('Processing and weighting style factor Earnings Quality')
    ProcessStyleFactorEarningsQuality(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Profitability
    print('Processing and weighting style factor Profitability')
    ProcessStyleFactorProfitability(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Investment Quality
    print('Processing and weighting style factor Investment Quality')
    ProcessStyleFactorInvestmentQuality(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Book-to-Price
    print('Processing and weighting style factor Book-to-Price')
    ProcessStyleFactorBooktoPrice(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Earnings Yield
    print('Processing and weighting style factor Earnings Yield')
    ProcessStyleFactorEarningsYield(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Long-Term Reversal
    print('Processing and weighting style factor Long-Term Reversal')
    ProcessStyleFactorLongTermReversal(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Growth
    print('Processing and weighting style factor Growth')
    ProcessStyleFactorGrowth(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Analyst Sentiment
    print('Processing and weighting style factor Analyst Sentiment')
    ProcessStyleFactorAnalystSentiment(weight_df=weight_df, I=I, path=path)
    # Process and weight style factor Dividend Yield
    print('Processing and weighting style factor Dividend Yield')
    ProcessStyleFactorDividendYield(weight_df=weight_df, I=I, path=path)
    return None


def Orthogonalization_Liquidity(factor):
    '''
    Orthogonalize Liquidity with respect to Size.
    '''
    factor['Liquidity_new'] = np.nan

    def cal_Liquidity_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20050912':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Size']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Liquidity']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Liquidity_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Liquidity")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Liquidity_new).reset_index(drop=True)
    factor = factor.drop('Liquidity', axis=1)
    factor.rename(columns={'Liquidity_new': 'Liquidity'}, inplace=True)
    return factor


def Orthogonalization_ResidualVolatility(factor):
    '''
    Orthogonalize Residual Volatility with respect to Beta and Size.
    '''
    factor['Residual Volatility_new'] = np.nan

    def cal_Residual_Volatility_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20070206':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, ['Beta', 'Size']]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Residual Volatility']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Residual Volatility_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Residual Volatility")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Residual_Volatility_new).reset_index(drop=True)
    factor = factor.drop('Residual Volatility', axis=1)
    factor.rename(columns={'Residual Volatility_new': 'Residual Volatility'}, inplace=True)
    return factor


def Orthogonalization_LongTermReversal(factor):
    '''
    Orthogonalize Long-Term Reversal with respect to Momentum.
    '''
    factor['Long-Term Reversal_new'] = np.nan

    def cal_Long_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, 'Momentum']
            Y = group[group['TRADE_DT'] == date].loc[:, 'Long-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Long-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Long-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Long_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Long-Term Reversal', axis=1)
    factor.rename(columns={'Long-Term Reversal_new': 'Long-Term Reversal'}, inplace=True)
    return factor


def Orthogonalization_AnalystSentiment(factor):
    '''
    Orthogonalize the short-term factor Analyst Sentiment with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Analyst Sentiment_new'] = np.nan

    def cal_Analyst_Sentiment_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':  # The latest starting date for the factor
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Analyst Sentiment']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Analyst Sentiment_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Analyst Sentiment")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Analyst_Sentiment_new).reset_index(drop=True)
    factor = factor.drop('Analyst Sentiment', axis=1)
    factor.rename(columns={'Analyst Sentiment_new': 'Analyst Sentiment'}, inplace=True)
    return factor


def Orthogonalization_IndustryMomentum(factor):
    '''
    Orthogonalize the short-term factor Industry Momentum with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Industry Momentum_new'] = np.nan

    def cal_Industry_Momentum_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Industry Momentum']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Industry Momentum_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Industry Momentum")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Industry_Momentum_new).reset_index(drop=True)
    factor = factor.drop('Industry Momentum', axis=1)
    factor.rename(columns={'Industry Momentum_new': 'Industry Momentum'}, inplace=True)
    return factor


def Orthogonalization_Seasonality(factor):
    '''
    Orthogonalize the short-term factor Seasonality with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Seasonality_new'] = np.nan

    def cal_Seasonality_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Seasonality']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Seasonality_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Seasonality")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Seasonality_new).reset_index(drop=True)
    factor = factor.drop('Seasonality', axis=1)
    factor.rename(columns={'Seasonality_new': 'Seasonality'}, inplace=True)
    return factor


def Orthogonalization_ShortTermReversal(factor):
    '''
    Orthogonalize the short-term factor Short-Term Reversal with respect to 16 long-term factors.
    '''
    longterm_factor_names = ["Beta", "Book-to-Price", "Dividend Yield", "Earnings Quality", "Earnings Variability",
                             "Earnings Yield", "Growth", "Investment Quality", "Leverage", "Liquidity", "Long-Term Reversal",
                             "Mid Capitalization", "Momentum", "Profitability", "Residual Volatility", "Size"]
    factor['Short-Term Reversal_new'] = np.nan

    def cal_Short_Term_Reversal_new(group):
        date = list(group["TRADE_DT"])[0]
        if date >= '20100528':
            X = group[group['TRADE_DT'] == date].loc[:, longterm_factor_names]
            Y = group[group['TRADE_DT'] == date].loc[:, 'Short-Term Reversal']
            X = X.fillna(X.median(axis=0))
            model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
            group.loc[model.resid.index, 'Short-Term Reversal_new'] = model.resid.values
        return group

    tqdm.pandas(desc="Orthogonalizing Short-Term Reversal")
    factor = factor.groupby('TRADE_DT').progress_apply(cal_Short_Term_Reversal_new).reset_index(drop=True)
    factor = factor.drop('Short-Term Reversal', axis=1)
    factor.rename(columns={'Short-Term Reversal_new': 'Short-Term Reversal'}, inplace=True)
    return factor


def ProcessFirstFactorSIZE(weight_df, path):
    '''
    Weight to obtain the first-level (CNLT) factor SIZE.
    '''
    Size = pd.read_csv(path + "风格因子\\" + "Size.csv", index_col=0)
    Mid_Capitalization = pd.read_csv(path + "风格因子\\" + "Mid Capitalization.csv", index_col=0)
    SIZE = 0.9 * Size - 0.1 * Mid_Capitalization

    SIZE = SIZE.stack().reset_index()
    SIZE = SIZE.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    SIZE = pd.merge(weight_df, SIZE, how='left')

    SIZE = standardize_factor(SIZE)
    SIZE_matrix = pd.pivot(SIZE, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    SIZE_matrix.to_csv(path + "CNLT大类因子\\" + "SIZE.csv")
    return None


def ProcessFirstFactorVOLATILITY(weight_df, path):
    '''
    Weight to obtain the first-level factor VOLATILITY.
    '''
    Beta = pd.read_csv(path + "风格因子\\" + "Beta.csv", index_col=0)
    Residual_Volatility = pd.read_csv(path + "风格因子\\" + "Residual Volatility.csv", index_col=0)
    VOLATILITY = 0.6 * Beta + 0.4 * Residual_Volatility

    VOLATILITY = VOLATILITY.stack().reset_index()
    VOLATILITY = VOLATILITY.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    VOLATILITY = pd.merge(weight_df, VOLATILITY, how='left')

    VOLATILITY = standardize_factor(VOLATILITY)
    VOLATILITY_matrix = pd.pivot(VOLATILITY, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VOLATILITY_matrix.to_csv(path + "CNLT大类因子\\" + "VOLATILITY.csv")
    return None


def ProcessFirstFactorLIQUIDITY(weight_df, path):
    '''
    Weight to obtain the first-level factor LIQUIDITY.
    '''
    Liquidity = pd.read_csv(path + "风格因子\\" + "Liquidity.csv", index_col=0)
    LIQUIDITY = Liquidity
    LIQUIDITY.to_csv(path + "CNLT大类因子\\" + "LIQUIDITY.csv")
    return None


def ProcessFirstFactorMOMENTUM(weight_df, path):
    '''
    Weight to obtain the first-level factor MOMENTUM.
    '''
    Momentum = pd.read_csv(path + "风格因子\\" + "Momentum.csv", index_col=0)
    MOMENTUM = Momentum
    MOMENTUM.to_csv(path + "CNLT大类因子\\" + "MOMENTUM.csv")
    return None


def ProcessFirstFactorQUALITY(weight_df, path):
    '''
    Weight to obtain the first-level factor QUALITY.
    '''
    Leverage = pd.read_csv(path + "风格因子\\" + "Leverage.csv", index_col=0)
    Earnings_Variability = pd.read_csv(path + "风格因子\\" + "Earnings Variability.csv", index_col=0)
    Earnings_Quality = pd.read_csv(path + "风格因子\\" + "Earnings Quality.csv", index_col=0)
    Profitability = pd.read_csv(path + "风格因子\\" + "Profitability.csv", index_col=0)
    Investment_Quality = pd.read_csv(path + "风格因子\\" + "Investment Quality.csv", index_col=0)
    QUALITY = -0.125 * Leverage - 0.125 * Earnings_Variability + 0.25 * Earnings_Quality + 0.25 * Profitability + 0.25 * Investment_Quality

    QUALITY = QUALITY.stack().reset_index()
    QUALITY = QUALITY.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    QUALITY = pd.merge(weight_df, QUALITY, how='left')

    QUALITY = standardize_factor(QUALITY)
    QUALITY_matrix = pd.pivot(QUALITY, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    QUALITY_matrix.to_csv(path + "CNLT大类因子\\" + "QUALITY.csv")
    return None


def ProcessFirstFactorVALUE(weight_df, path):
    '''
    Weight to obtain the first-level factor VALUE.
    '''
    Book_to_Price = pd.read_csv(path + "风格因子\\" + "Book-to-Price.csv", index_col=0)
    Earnings_Yield = pd.read_csv(path + "风格因子\\" + "Earnings Yield.csv", index_col=0)
    Long_Term_Reversal = pd.read_csv(path + "风格因子\\" + "Long-Term Reversal.csv", index_col=0)

    VALUE = 0.3 * Book_to_Price + 0.6 * Earnings_Yield + 0.1 * Long_Term_Reversal

    VALUE = VALUE.stack().reset_index()
    VALUE = VALUE.rename(columns={'level_1': 'TRADE_DT', 0: 'FACTOR'})
    VALUE = pd.merge(weight_df, VALUE, how='left')

    VALUE = standardize_factor(VALUE)
    VALUE_matrix = pd.pivot(VALUE, index='S_INFO_WINDCODE', columns='TRADE_DT', values='FACTOR')
    VALUE_matrix.to_csv(path + "CNLT大类因子\\" + "VALUE.csv")
    return None


def ProcessFirstFactorGROWTH(weight_df, path):
    '''
    Weight to obtain the first-level factor GROWTH.
    '''
    Growth = pd.read_csv(path + "风格因子\\" + "Growth.csv", index_col=0)
    GROWTH = Growth
    GROWTH.to_csv(path + "CNLT大类因子\\" + "GROWTH.csv")
    return None


def ProcessFirstFactorYIELD(weight_df, path):
    '''
    Weight to obtain the first-level factor YIELD.
    '''
    Dividend_Yield = pd.read_csv(path + "风格因子\\" + "Dividend Yield.csv", index_col=0)
    YIELD = Dividend_Yield
    YIELD.to_csv(path + "CNLT大类因子\\" + "YIELD.csv")
    return None


def ProcessFirstFactor(weight_df, path):
    '''
    Weight to obtain all first-level (CNLT) factors.
    '''
    if not os.path.exists(path + "CNLT大类因子\\"):
        os.makedirs(path + "CNLT大类因子\\")
    print('Weighting first-level factor SIZE')
    ProcessFirstFactorSIZE(weight_df=weight_df, path=path)
    print('Weighting first-level factor VOLATILITY')
    ProcessFirstFactorVOLATILITY(weight_df=weight_df, path=path)
    print('Weighting first-level factor LIQUIDITY')
    ProcessFirstFactorLIQUIDITY(weight_df=weight_df, path=path)
    print('Weighting first-level factor MOMENTUM')
    ProcessFirstFactorMOMENTUM(weight_df=weight_df, path=path)
    print('Weighting first-level factor QUALITY')
    ProcessFirstFactorQUALITY(weight_df=weight_df, path=path)
    print('Weighting first-level factor VALUE')
    ProcessFirstFactorVALUE(weight_df=weight_df, path=path)
    print('Weighting first-level factor GROWTH')
    ProcessFirstFactorGROWTH(weight_df=weight_df, path=path)
    print('Weighting first-level factor YIELD')
    ProcessFirstFactorYIELD(weight_df=weight_df, path=path)

    return None


if __name__ == "__main__":
    # Start and end dates
    start_date = '20050104'
    end_date = '20240524'
    # Set the path for factor data
    path = "F:\\因子数据\\"
    # Get base data: all trading days and tradable stocks
    BASE = getAShareBase(start_date, end_date)
    # Read the market cap weight and industry data required for factor processing
    weight_df, I = GetProcessData(start_date=start_date, end_date=end_date, BASE=BASE)

    # Generate industry and country factors (uncomment if needed)
    CreatIndustryCountry(BASE=BASE, I=I, path=path)
    # Process and weight style factors
    ProcessStyleFactor(weight_df=weight_df, I=I, path=path)
    # Orthogonalize style factors
    Orthogonalization(BASE=BASE, path=path)
    # Weight to obtain first-level (CNLT) factors
    ProcessFirstFactor(weight_df=weight_df, path=path)

