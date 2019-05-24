__author__ = 'Cila'
import pyodbc
import pandas as pd
import math
import datetime as dt

database = "Bitcoin"
host = "192.168.200.11"
user = "sa"
pwd = "sa123$%^"
startDate = "2018-09-10"
endDate = "2018-09-30"

#计算平均数
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


connInfo = 'DRIVER={SQL Server};DATABASE=%s;SERVER=%s;UID=%s;PWD=%s' % (database, host, user, pwd)
msSqlConn = pyodbc.connect(connInfo)

startDt = dt.datetime.strptime(startDate, '%Y-%m-%d')
endDt = dt.datetime.strptime(endDate, '%Y-%m-%d')
days = (endDt - startDt).days + 1
newTimeStamp = 0
newTickData = []
tempTickData = []
for k in range(days):
    nextDay = startDt + dt.timedelta(days=1)
    nextDayStr = nextDay.strftime('%Y-%m-%d')
    sql = "select * from tick_data_bitmex where tick_time>='" + startDate + "'and tick_time<'" + nextDayStr + "' and tick_code = 'XBTUSD' order by tick_time asc"
    df = pd.read_sql(sql, msSqlConn)
    rawData = pd.DataFrame(df).values

    for i in range(len(rawData)):
        tickTime = math.floor(rawData[i][3] / 10000) * 10 + 10
        if newTimeStamp != tickTime:
            if newTimeStamp != 0:
                tickData = []
                tickData.append(newTimeStamp)
                tickData.append(averagenum(tempTickData))
                newTickData.append(tickData)
                tempTickData = []
            newTimeStamp = tickTime
        askPrice = rawData[i][4]
        bidPrice = rawData[i][24]
        lastPrice = (askPrice + bidPrice) / 2
        tempTickData.append(lastPrice)
    startDate = nextDayStr
    startDt = nextDay
writer = pd.ExcelWriter("../Data/RawData10s.xlsx")
writeData = pd.DataFrame(newTickData,columns=['ts','price'])
writeData.to_excel(writer, index=False)
writer.save()
# labelData = []
# for i in range(len(newTickData)):
#     if i < len(newTickData):
#         if newTickData[i][1]