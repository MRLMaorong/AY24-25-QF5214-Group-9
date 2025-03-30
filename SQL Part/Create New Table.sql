USE [zyyx0524]
GO

/****** Object:  Table [dbo].[eps_latest_report]    Script Date: 10/26/2023 12:47:04 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING ON
GO


CREATE TABLE zyyx0524.dbo.trade_days (
	[tradedate] [varchar](20) NULL
) ON [PRIMARY]



CREATE TABLE zyyx0524.dbo.pe_latest_report (
	[tradedate] [date] NOT NULL,
	[report_id] [int] NOT Null,
	[stock_code] [varchar](20) NULL,
	[organ_id] [int] Null,
	[create_date] [date] Null,
	[entrydate] [date] Null,
	[report_year] [int] Null,
	[forecast_pe] [float] Null,
	[fy] [int] NOT Null
) ON [PRIMARY]

CREATE TABLE zyyx0524.dbo.eps_latest_report (
	[tradedate] [date] NOT NULL,
	[report_id] [int] NOT Null,
	[stock_code] [varchar](20) NULL,
	[organ_id] [int] Null,
	[create_date] [date] Null,
	[entrydate] [date] Null,
	[report_year] [int] Null,
	[forecast_eps] [float] Null,
	[fy] [int] NOT Null
) ON [PRIMARY]

CREATE TABLE zyyx0524.dbo.dps_latest_report (
	[tradedate] [date] NOT NULL,
	[report_id] [int] NOT Null,
	[stock_code] [varchar](20) NULL,
	[organ_id] [int] Null,
	[create_date] [date] Null,
	[entrydate] [date] Null,
	[report_year] [int] Null,
	[forecast_dps] [float] Null,
	[fy] [int] NOT Null
) ON [PRIMARY]

CREATE TABLE zyyx0524.dbo.np_fy1 (
	[tradedate] [date] NOT NULL,
	[report_id] [int] NOT Null,
	[stock_code] [varchar](20) NULL,
	[organ_id] [int] Null,
	[create_date] [date] Null,
	[entrydate] [date] Null,
	[report_year] [int] Null,
	[np_fy1] [float] Null,
	[fy] [int] NOT Null
) ON [PRIMARY]

CREATE TABLE zyyx0524.dbo.np_fy2 (
	[tradedate] [date] NOT NULL,
	[report_id] [int] NOT Null,
	[stock_code] [varchar](20) NULL,
	[organ_id] [int] Null,
	[create_date] [date] Null,
	[entrydate] [date] Null,
	[report_year] [int] Null,
	[np_fy2] [float] Null,
	[fy] [int] NOT Null
) ON [PRIMARY]

CREATE TABLE zyyx0524.dbo.np_fy3 (
	[tradedate] [date] NOT NULL,
	[report_id] [int] NOT Null,
	[stock_code] [varchar](20) NULL,
	[organ_id] [int] Null,
	[create_date] [date] Null,
	[entrydate] [date] Null,
	[report_year] [int] Null,
	[np_fy3] [float] Null,
	[fy] [int] NOT Null
) ON [PRIMARY]


GO

SET ANSI_PADDING OFF
GO


