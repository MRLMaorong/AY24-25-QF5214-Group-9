# -*- coding: utf-8 -*-
"""
Created in 2025
Function : Load SQL
"""
import warnings
import connectorx as cx
import pymssql

warnings.filterwarnings(action='ignore')
from Tools import *

# Connect to SQL Server using pymssql
# Wind database
db_wind = pymssql.connect(host='127.0.0.1:50692', user='sa', password='##########', database='winddb0524')
# Chaoyang Yongxu database
db_zyyx = pymssql.connect(host='127.0.0.1:50692', user='sa', password='##########', database='zyyx0524')

# Use connectorx for acceleration
URI_db_wind = "mssql+pyodbc://sa:##########@127.0.0.1:50692/winddb0524"
URI_db_zyyx = "mssql+pyodbc://sa:##########@127.0.0.1:50692/zyyx0524"

def getAShareCalender():
    """
    Function Name: getAShareCalender
    Functionality: Reads the ASHARECALENDAR table.
    Input Parameters: None
    Output: ASHARECALENDAR table data.
    """
    sql = "select TRADE_DAYS from ASHARECALENDAR ORDER BY TRADE_DAYS"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    df.drop_duplicates(inplace=True)
    return df


def getAShareEODDerivativeIndicator(start_date, end_date):
    """
    Function Name: getAShareEODDerivativeIndicator
    Functionality: Reads the ASHAREEODDERIVATIVEINDICATOR table.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHAREEODDERIVATIVEINDICATOR table data.
    """
    sql = "select S_INFO_WINDCODE,TRADE_DT,S_VAL_MV,S_DQ_MV,S_VAL_PB_NEW,S_DQ_FREETURNOVER,FREE_SHARES_TODAY*S_DQ_CLOSE_TODAY as FREE_FLOAT_CAP from ASHAREEODDERIVATIVEINDICATOR WHERE TRADE_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' order by S_INFO_WINDCODE,TRADE_DT"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareEODPrices(start_date, end_date):
    """
    Function Name: getAShareEODPrices
    Functionality: Reads the ASHAREEODPRICES table.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHAREEODPRICES table data.
    """
    sql = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE,S_DQ_PRECLOSE,S_DQ_VOLUME,S_DQ_ADJCLOSE,S_DQ_ADJPRECLOSE from ASHAREEODPRICES where TRADE_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%'order by S_INFO_WINDCODE,TRADE_DT"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAindexEODPrices(start_date, end_date):
    """
    Function Name: getAindexEODPrices
    Functionality: Reads the AINDEXEODPRICES table.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: AINDEXEODPRICES table data.
    """
    sql = "select S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE,S_DQ_PRECLOSE from AINDEXEODPRICES where TRADE_DT between \'" + start_date + "\' and \'" + end_date + "\' order by S_INFO_WINDCODE,TRADE_DT"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareBase(start_date, end_date):
    """
    Function Name: getAShareBase
    Functionality: Reads all tradable dates and stocks from the ASHAREEODPRICES table to serve as the base data for the factor model.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: Base data table.
    """
    sql = "select S_INFO_WINDCODE,TRADE_DT from ASHAREEODPRICES WHERE TRADE_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,TRADE_DT"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareIndustries_old(start_date, end_date):
    """
    Function Name: getAShareIndustries
    Functionality: Based on the ASHAREEODPRICES table, reads the industries each stock belongs to.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: Industry data for each stock.
    """
    # For columns containing Chinese characters, perform format conversion (convert(nvarchar(50), t2.INDUSTRIESNAME) as 'INDUSTRIESNAME') to avoid errors
    sql = "select t1.S_INFO_WINDCODE,t1.TRADE_DT,convert(nvarchar(50),t2.INDUSTRIESNAME) as 'INDUSTRIESNAME' from (select S_INFO_WINDCODE,TRADE_DT from ASHAREEODPRICES WHERE TRADE_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%'and TRADE_DT in (select TRADE_DAYS from AShareCalendar)) as t1 left join (select aa.s_info_windcode,bb.Industriesname from AShareIndustriesClassCITICS aa,AShareIndustriesCode  bb where substring(aa.citics_ind_code, 1, 4) = substring(bb.IndustriesCode, 1, 4)and bb.levelnum = '2' and aa.cur_sign = '1') t2 on t1.S_INFO_WINDCODE = t2.s_info_windcode order by t1.S_INFO_WINDCODE,t1.TRADE_DT"
    df = pd.read_sql(sql, db_wind)
    # df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareIndustriesClass():
    """
    Function Name: getAShareIndustriesClass
    Functionality: Reads changes in stock industry classification information and further processes it to obtain time-series data.
    Input Parameters: (None specified)
    Output: Industry classification change data for each stock.
    """
    # For columns containing Chinese characters, perform format conversion to avoid errors
    sql = "select a.S_INFO_WINDCODE,a.ENTRY_DT,a.REMOVE_DT,a.CUR_SIGN,convert(nvarchar(50),b.Industriesname) as 'INDUSTRIESNAME' from AShareIndustriesClassCITICS a,AShareIndustriesCode b where substring(a.citics_ind_code, 1, 4) = substring(b.IndustriesCode, 1, 4) and b.levelnum = '2'"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareListDate():
    """
    Function Name: getAShareListDate
    Functionality: Reads the listing dates for each stock from the ASHAREDESCRIPTION table.
    Input Parameters: None
    Output: Listing dates for each stock.
    """
    sql = "select S_INFO_WINDCODE,S_INFO_LISTDATE from ASHAREDESCRIPTION where S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareIncome(start_date, end_date):
    """
    Function Name: getAShareIncome
    Functionality: Reads the ASHAREINCOME table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHAREINCOME table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE,report_period as REPORT_PERIOD,ANN_DT,OPER_REV,TOT_OPER_REV,TOT_OPER_COST,NET_PROFIT_AFTER_DED_NR_LP,EBIT from ASHAREINCOME where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and statement_type = '408001000'and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareBalanceSheet(start_date, end_date):
    """
    Function Name: getAShareBalanceSheet
    Functionality: Reads the ASHAREBALANCESHEET table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHAREBALANCESHEET table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD,TOT_NON_CUR_LIAB,TOT_SHRHLDR_EQY_EXCL_MIN_INT,TOT_ASSETS,TOT_LIAB,OTHER_EQUITY_TOOLS_P_SHR from ASHAREBALANCESHEET where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and statement_type = '408001000' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareCashFlow(start_date, end_date):
    """
    Function Name: getAShareCashFlow
    Functionality: Reads the ASHARECASHFLOW table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHARECASHFLOW table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD,NET_INCR_CASH_CASH_EQU,NET_PROFIT,NET_CASH_FLOWS_OPER_ACT,NET_CASH_FLOWS_INV_ACT,CASH_CASH_EQU_END_PERIOD,CASH_PAY_ACQ_CONST_FIOLTA from ASHARECASHFLOW where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and statement_type = '408001000' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareFinancialIndicator(start_date, end_date):
    """
    Function Name: getAShareFinancialIndicator
    Functionality: Reads the ASHAREFINANCIALINDICATOR table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHAREFINANCIALINDICATOR table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD,S_FA_DEBTTOASSETS,S_STM_IS,S_FA_ROA,S_FA_NETDEBT,S_FA_INTERESTDEBT,S_FA_ASSETSTURN,S_FA_GROSSPROFITMARGIN,S_FA_CAPITALIZEDTODA,S_FA_EBIT,S_FA_EXINTERESTDEBT_CURRENT,S_FA_EXINTERESTDEBT_NONCURRENT from ASHAREFINANCIALINDICATOR where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ'and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareFreeFloat(start_date, end_date):
    """
    Function Name: getAShareFreeFloat
    Functionality: Reads the ASHAREFREEFLOAT table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHAREFREEFLOAT table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE,ANN_DT,S_SHARE_FREESHARES from ASHAREFREEFLOAT where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ'and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareDividend(start_date, end_date):
    """
    Function Name: getAShareDividend
    Functionality: Reads the ASHAREDIVIDEND table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHAREDIVIDEND table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD,CASH_DVD_PER_SH_AFTER_TAX,S_DIV_PROGRESS from ASHAREDIVIDEND where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ'and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareTTMHis(start_date, end_date):
    """
    Function Name: getAShareTTMHis
    Functionality: Reads the ASHARETTMHIS table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHARETTMHIS table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD,OPER_REV_TTM,NET_PROFIT_TTM,NET_CASH_FLOWS_OPER_ACT_TTM from ASHARETTMHIS where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,ANN_DT,REPORT_PERIOD"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareCapitalization(start_date, end_date):
    """
    Function Name: getAShareCapitalization
    Functionality: Reads the ASHARECAPITALIZATION table and preprocesses the ANN_DT field.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: ASHARECAPITALIZATION table data with preprocessed ANN_DT.
    """
    sql = "SELECT S_INFO_WINDCODE, CHANGE_DT, ANN_DT, TOT_SHR,FLOAT_SHR,NON_TRADABLE_SHR, S_SHARE_CHANGEREASON FROM ASHARECAPITALIZATION where ANN_DT between \'" + start_date + "\' and \'" + end_date + "\' and S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE,CHANGE_DT,ANN_DT"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    # Preprocess ANN_DT: move ANN_DT on non-trading days to the next trading day
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    return df


def getAShareTypeCode():
    """
    Function Name: getAShareTypeCode
    Functionality: Reads the ASHARETYPECODE table.
    Input Parameters: None
    Output: ASHARETYPECODE table data.
    """
    # For columns containing Chinese characters, perform format conversion to avoid errors
    sql = "select convert(nvarchar(50),S_TYPNAME) as 'S_TYPNAME',S_TYPCODE from ASHARETYPECODE where convert(nvarchar(50),S_CLASSIFICATION) = '股本变动原因代码表'"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareListDate():
    """
    Function Name: getAShareListDate
    Functionality: Retrieves stock listing dates from the ASHAREDESCRIPTION table.
    Input Parameters: None
    Output: Stock listing date data table.
    """
    sql = "select S_INFO_WINDCODE,S_INFO_LISTDATE from ASHAREDESCRIPTION where S_INFO_WINDCODE not like '%BJ' and S_INFO_WINDCODE not like 'T%' and S_INFO_WINDCODE not like 'A%' order by S_INFO_WINDCODE"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getAShareIndexWeight(start_date, end_date):
    """
    Function Name: getAShareIndexWeight
    Functionality: Retrieves index constituent stock weights from the AINDEXHS300FREEWEIGHT table.
    Input Parameters:
        start_date: Start date
        end_date: End date
    Output: Index constituent weights.
    """
    sql = "select S_INFO_WINDCODE,S_CON_WINDCODE,TRADE_DT,I_WEIGHT from AINDEXHS300FREEWEIGHT where  TRADE_DT between \'" + start_date + "\' and \'" + end_date + "\'  order by S_INFO_WINDCODE,TRADE_DT,S_CON_WINDCODE"
    # df = pd.read_sql(sql, db_wind)
    df = cx.read_sql(URI_db_wind, sql, partition_num=10)  # Use connectorx for acceleration
    return df


def getForecastEP(start_date, end_date, tickerlist):
    """
    Function Name: getForecastEP
    Functionality: Reads the Chaoyang Yongxu pe_latest_report table (pre-processed in SQL), calculates the reciprocal, and adjusts the data format to be consistent with the WIND database.
    Input Parameters:
        start_date: Start date
        end_date: End date
        tickerlist: List of all stock codes, for data format processing.
    Output: Analyst forecast EP data.
    """
    sql = "SELECT tradedate,stock_code,forecast_pe FROM pe_latest_report where tradedate between \'" + start_date + "\' and \'" + end_date + "\' ORDER BY stock_code,tradedate"
    # df = pd.read_sql(sql, db_zyyx)
    df = cx.read_sql(URI_db_zyyx, sql, partition_num=10)  # Use connectorx for acceleration
    # Calculate reciprocal
    df['forecast_ep'] = 1 / df['forecast_pe']
    df = df.drop(columns=['forecast_pe'])
    # Rename columns to match the WIND database
    df.rename(columns={'tradedate': 'TRADE_DT', 'stock_code': 'S_INFO_WINDCODE'}, inplace=True)

    # Adjust time format to be consistent with the WIND database
    def time_format(t):
        timestr = str(t)[0:10]
        return timestr[0:4] + timestr[5:7] + timestr[8:10]

    df['TRADE_DT'] = df['TRADE_DT'].map(time_format)
    # Preprocess S_INFO_WINDCODE
    df = Process_Stock_Code(df, tickerlist)
    # Remove rows with NaN values due to stock code conversion issues (e.g., Beijing Stock Exchange, B-shares)
    df = df.dropna()
    return df


def getForecastEPS(start_date, end_date, tickerlist):
    """
    Function Name: getForecastEPS
    Functionality: Reads the Chaoyang Yongxu eps_latest_report table (pre-processed in SQL) and adjusts the data format to be consistent with the WIND database.
    Input Parameters:
        start_date: Start date
        end_date: End date
        tickerlist: List of all stock codes, for data format processing.
    Output: Analyst forecast EPS data.
    """
    sql = "SELECT tradedate,stock_code,forecast_eps from eps_latest_report where tradedate between \'" + start_date + "\' and \'" + end_date + "\' ORDER BY stock_code,tradedate"
    df = pd.read_sql(sql, db_zyyx)
    # df = cx.read_sql(URI_db_zyyx, sql, partition_num=10)  # Use connectorx for acceleration
    # Rename columns to match the WIND database
    df.rename(columns={'tradedate': 'TRADE_DT', 'stock_code': 'S_INFO_WINDCODE'}, inplace=True)

    # Adjust time format to be consistent with the WIND database
    def time_format(t):
        timestr = str(t)[0:10]
        return timestr[0:4] + timestr[5:7] + timestr[8:10]

    df['TRADE_DT'] = df['TRADE_DT'].map(time_format)
    # Preprocess S_INFO_WINDCODE
    df = Process_Stock_Code(df, tickerlist)
    # Remove rows with NaN values due to stock code conversion issues (e.g., Beijing Stock Exchange, B-shares)
    df = df.dropna()
    return df


def getForecastDPS(start_date, end_date, tickerlist):
    """
    Function Name: getForecastDPS
    Functionality: Reads the Chaoyang Yongxu dps_latest_report (pre-processed in SQL) and adjusts the data format to be consistent with the WIND database.
    Input Parameters:
        start_date: Start date
        end_date: End date
        tickerlist: List of all stock codes, for data format processing.
    Output: Analyst forecast DPS data.
    """
    sql = "SELECT tradedate,stock_code,forecast_dps from dps_latest_report where tradedate between \'" + start_date + "\' and \'" + end_date + "\' ORDER BY stock_code,tradedate"
    # df = pd.read_sql(sql, db_zyyx)
    df = cx.read_sql(URI_db_zyyx, sql, partition_num=10)  # Use connectorx for acceleration
    # Rename columns to match the WIND database
    df.rename(columns={'tradedate': 'TRADE_DT', 'stock_code': 'S_INFO_WINDCODE'}, inplace=True)

    # Adjust time format to be consistent with the WIND database
    def time_format(t):
        timestr = str(t)[0:10]
        return timestr[0:4] + timestr[5:7] + timestr[8:10]

    df['TRADE_DT'] = df['TRADE_DT'].map(time_format)
    # Preprocess S_INFO_WINDCODE
    df = Process_Stock_Code(df, tickerlist)
    # Remove rows with NaN values due to stock code conversion issues (e.g., Beijing Stock Exchange, B-shares)
    df = df.dropna()
    return df


def getForecastEarningsAdjust(start_date, end_date, tickerlist):
    """
    Function Name: getForecastEarningsAdjust
    Functionality: Reads the Chaoyang Yongxu rpt_earnings_adjust (pre-processed in SQL) and adjusts the data format to be consistent with the WIND database.
    Input Parameters:
        start_date: Start date
        end_date: End date
        tickerlist: List of all stock codes, for data format processing.
    Output: Analyst earnings adjustment data.
    """
    sql = "SELECT entrytime,stock_code,np_adjust_mark,report_year from rpt_earnings_adjust where entrytime between \'" + start_date + "\' and \'" + end_date + "\' ORDER BY stock_code,entrytime"
    # df = pd.read_sql(sql, db_zyyx)
    df = cx.read_sql(URI_db_zyyx, sql, partition_num=10)  # Use connectorx for acceleration
    # Adjust numerical values: 1 -> 0; 2 -> 1; 3 -> -1
    df['np_adjust_mark'] = df['np_adjust_mark'].replace(1, 0)
    df['np_adjust_mark'] = df['np_adjust_mark'].replace(2, 1)
    df['np_adjust_mark'] = df['np_adjust_mark'].replace(3, -1)

    # Adjust time format to be consistent with the WIND database
    def time_format(t):
        timestr = str(t)[0:10]
        return timestr[0:4] + timestr[5:7] + timestr[8:10]

    df['entrytime'] = df['entrytime'].map(time_format)
    # Only keep fy1, i.e., if entrytime.month <= 4 and entrytime.year - 1 equals report_year, or if entrytime.month > 4 and entrytime.year equals report_year, then it's fy1
    df['fy'] = [1 if int(df['entrytime'].values[i][4:6]) <= 4 and int(df['entrytime'].values[i][:4]) - 1 == df['report_year'].values[i] or
                           int(df['entrytime'].values[i][4:6]) > 4 and int(df['entrytime'].values[i][:4]) == df['report_year'].values[i] else -1 for i in range(len(df))]
    df = df[df['fy'] == 1]
    # Rename columns to match the WIND database
    df.rename(columns={'entrytime': 'ANN_DT', 'stock_code': 'S_INFO_WINDCODE'}, inplace=True)
    # Preprocess ANN_DT
    AShareCalender = getAShareCalender()
    df = Process_ANN_DT(df, AShareCalender)
    # Preprocess S_INFO_WINDCODE
    df = Process_Stock_Code(df, tickerlist)
    # Remove records without investment ratings and rows with NaN values due to stock code conversion issues (e.g., Beijing Stock Exchange, B-shares)
    df = df.dropna()
    return df


def getforecastNP_FY1(start_date, end_date, tickerlist):
    """
    Function Name: getforecastNP_FY1
    Functionality: Reads the Chaoyang Yongxu np_fy1 (pre-processed in SQL) and adjusts the data format to be consistent with the WIND database.
    Input Parameters:
        start_date: Start date
        end_date: End date
        tickerlist: List of all stock codes, for data format processing.
    Output: Analyst np_fy1 data.
    """
    sql = "SELECT tradedate,stock_code,np_fy1 from np_fy1 where tradedate between \'" + start_date + "\' and \'" + end_date + "\' ORDER BY stock_code,tradedate"
    # df = pd.read_sql(sql, db_zyyx)
    df = cx.read_sql(URI_db_zyyx, sql, partition_num=10)  # Use connectorx for acceleration
    # Rename columns to match the WIND database
    df.rename(columns={'tradedate': 'TRADE_DT', 'stock_code': 'S_INFO_WINDCODE'}, inplace=True)

    # Adjust time format to be consistent with the WIND database
    def time_format(t):
        timestr = str(t)[0:10]
        return timestr[0:4] + timestr[5:7] + timestr[8:10]

    df['TRADE_DT'] = df['TRADE_DT'].map(time_format)
    # Preprocess S_INFO_WINDCODE
    df = Process_Stock_Code(df, tickerlist)
    # Remove rows with NaN values due to stock code conversion issues (e.g., Beijing Stock Exchange, B-shares)
    df = df.dropna()
    return df


def getforecastNP_FY2(start_date, end_date, tickerlist):
    """
    Function Name: getforecastNP_FY2
    Functionality: Reads the Chaoyang Yongxu np_fy2 (pre-processed in SQL) and adjusts the data format to be consistent with the WIND database.
    Input Parameters:
        start_date: Start date
        end_date: End date
        tickerlist: List of all stock codes, for data format processing.
    Output: Analyst np_fy2 data.
    """
    sql = "SELECT tradedate,stock_code,np_fy2 from np_fy2 where tradedate between \'" + start_date + "\' and \'" + end_date + "\' ORDER BY stock_code,tradedate"
    # df = pd.read_sql(sql, db_zyyx)
    df = cx.read_sql(URI_db_zyyx, sql, partition_num=10)  # Use connectorx for acceleration
    # Rename columns to match the WIND database
    df.rename(columns={'tradedate': 'TRADE_DT', 'stock_code': 'S_INFO_WINDCODE'}, inplace=True)

    # Adjust time format to be consistent with the WIND database
    def time_format(t):
        timestr = str(t)[0:10]
        return timestr[0:4] + timestr[5:7] + timestr[8:10]

    df['TRADE_DT'] = df['TRADE_DT'].map(time_format)
    # Preprocess S_INFO_WINDCODE
    df = Process_Stock_Code(df, tickerlist)
    # Remove rows with NaN values due to stock code conversion issues (e.g., Beijing Stock Exchange, B-shares)
    df = df.dropna()
    return df


def getforecastNP_FY3(start_date, end_date, tickerlist):
    """
    Function Name: getforecastNP_FY3
    Functionality: Reads the Chaoyang Yongxu np_fy3 (pre-processed in SQL) and adjusts the data format to be consistent with the WIND database.
    Input Parameters:
        start_date: Start date
        end_date: End date
        tickerlist: List of all stock codes, for data format processing.
    Output: Analyst np_fy3 data.
    """
    sql = "SELECT tradedate,stock_code,np_fy3 from np_fy3 where tradedate between \'" + start_date + "\' and \'" + end_date + "\' ORDER BY stock_code,tradedate"
    # df = pd.read_sql(sql, db_zyyx)
    df = cx.read_sql(URI_db_zyyx, sql, partition_num=10)  # Use connectorx for acceleration
    # Rename columns to match the WIND database
    df.rename(columns={'tradedate': 'TRADE_DT', 'stock_code': 'S_INFO_WINDCODE'}, inplace=True)

    # Adjust time format to be consistent with the WIND database
    def time_format(t):
        timestr = str(t)[0:10]
        return timestr[0:4] + timestr[5:7] + timestr[8:10]

    df['TRADE_DT'] = df['TRADE_DT'].map(time_format)
    # Preprocess S_INFO_WINDCODE
    df = Process_Stock_Code(df, tickerlist)
    # Remove rows with NaN values due to stock code conversion issues (e.g., Beijing Stock Exchange, B-shares)
    df = df.dropna()
    return df
