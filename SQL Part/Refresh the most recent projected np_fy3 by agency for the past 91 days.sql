declare @tradingDay varchar(8) = '20050104'
WHILE @tradingDay <= '20240524'

BEGIN
	with
	cr1 as
	(
		select report_id, stock_code, organ_id, create_date, CAST(entrytime as date) as entrydate,
		report_year, forecast_np,
		ROW_NUMBER() over (
		partition by report_id order by report_year asc) as fy
		from zyyx0524.dbo.rpt_forecast_stk
		where DATEDIFF(DAY, create_date, CAST(entrytime as date)) <= 7
		and CAST(entrytime as date) >= DATEADD(day,-91,cast(@tradingDay as date))
		and CAST(entrytime as date) <= @tradingDay
		and forecast_np is not null
		and report_quarter = 4
		and (reliability is null or reliability > 4)
	),
	cr2 as
	(
		select report_id, stock_code, organ_id, create_date, entrydate,
		report_year, forecast_np, fy, ROW_NUMBER() over(
		partition by stock_code, organ_id, report_year order by create_date desc, entrydate desc) as seq
		from cr1
		where fy = 3
	),
	cr3 as(
		select report_id, stock_code, organ_id, create_date, entrydate, report_year, forecast_np, fy 
		from cr2
		where
		(case when MONTH(@tradingDay) <= 4 then YEAR(@tradingDay)+1 else YEAR(@tradingDay)+2 end) = report_year
		and seq = 1
	)
	insert into zyyx0524.dbo.np_fy3(tradedate,report_id,stock_code, organ_id, create_date, entrydate, report_year, np_fy3, fy)
	select @tradingDay, report_id, stock_code, organ_id, create_date, entrydate, report_year, forecast_np, fy
	from cr3
	order by stock_code, organ_id
	
	select @tradingDay = b.tradedate from
	(select TOP 1 a.tradedate from zyyx0524.dbo.trade_days a where a.tradedate > @tradingDay
	order by a.tradedate) b
	
	print @tradingDay
	
END