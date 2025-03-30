insert into zyyx0524.dbo.trade_days (tradedate)
select distinct TRADE_DAYS
from winddb0524.dbo.AShareCalendar
where TRADE_DAYS >= '20050101' and TRADE_DAYS <= '20241231'
order by TRADE_DAYS
