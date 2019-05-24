__author__ = 'Cila'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymssql
import math
import datetime
import GetCoinRate
import numpy as np
import pandas as pd
import os

Holidays = {
               '20171230':4,
               '20180215':8,
               '20180405':5,
               '20180428':5,
               '20180616':4,
               '20180922':4,
               '20180929':10
            }
conn = pymssql.connect("127.0.0.1","sa","sa123$%^","StockPolicy")
cur = conn.cursor()
trade_time = '2018-05-30'
jingzhi = 1
account = '40573476'
per_capital = [4600000]
policys = ['Policy511880']
policy_gain = []
policy_jiaoyi = []
policy_chicang = []
policy_nihuigou = []
policy_date = []

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        if root == file_dir:
            for file in files:
                if os.path.splitext(file)[1] == '.xlsx':
                    L.append(os.path.join(root, file))
    return L

def AddDays(date,days):
    return (date+datetime.timedelta(days = days)).strftime('%Y%m%d')

writer = pd.ExcelWriter(account + '/货基收益明细%' + account + '%' + trade_time + '.xlsx')
for policy_index,policy in enumerate(policys):
    cur.execute("select * from trade_hist where trade_policyname = '" + policy + "' and trade_time > '"+ trade_time +"' and trade_account = '" + account + "' order by trade_time asc")
    data = cur.fetchall()
    gain = []
    nihuigou = []
    chicangguoye = []
    jiaoyi = []
    dates = []
    dates.append(trade_time)
    temp_time = datetime.datetime.strptime(trade_time,'%Y-%m-%d').strftime('%Y-%m-%d')
    total = 0
    jy = 0
    cc = 0
    nhg = 0
    gc = GetCoinRate.GetCoinRate()
    tick_rate = gc.coinRate(policy[6:],trade_time)
    tick_519888 = gc.coinRate('519888',trade_time)
    tick_001621 = gc.coinRate('001621',trade_time)
    tick_004796 = gc.coinRate('004796',trade_time)
    tick_000540 = gc.coinRate('000540',trade_time)
    for i,temp in enumerate(data):
        data_time = temp[20].strftime('%Y-%m-%d')
        if data_time > temp_time:
            gain.append(total)
            jiaoyi.append(jy)
            chicangguoye.append(cc)
            nihuigou.append(nhg)
            jy = 0
            cc = 0
            nhg = 0
            total = 0
            temp_time = data_time
            if data_time not in dates:
                dates.append(data_time)
        if temp[2] == '131810' or temp[2] == '204001':
            if temp[11] != 0:
                nhg_lirun = float(temp[11]) / 365 * temp[12]
                sign_day = AddDays(temp[20],2)
                if Holidays.__contains__(sign_day):
                    nhg_lirun = nhg_lirun * Holidays[sign_day]
                if temp[20].strftime('%w') == '4' and not Holidays.__contains__(AddDays(temp[20],1)) and not Holidays.__contains__(AddDays(temp[20],2)):
                    nhg_lirun = nhg_lirun * 3
                nhg_lirun -= (temp[12] / 100 * 0.1)
                nhg += nhg_lirun
                total += nhg_lirun
        elif temp[2] == '519888':
            dt = temp[20].strftime('%Y-%m-%d')
            index = np.where(dt == tick_519888)
            if len(index[0]) > 0:
                cc_lirun = 0
                index = index[0][0]
                cc_lirun = tick_519888[index][3] * temp[12] / 10e+6
                if temp[20].strftime('%w') == '5' or Holidays.__contains__(AddDays(temp[20],1)):
                    if len(tick_rate) > index + 1:
                        cc_lirun += tick_519888[index+1][3] * temp[12] / 10e+6
                cc += cc_lirun
                total += cc_lirun
        elif temp[2] == '001621' or temp[2] == '004796' or temp[2] == '000540':
            cc_lirun = 0
            dt = (temp[20]+datetime.timedelta(days = 1)).strftime('%Y-%m-%d')
            if temp[2] == '001621':
                rate = tick_001621
            elif temp[2] == '004796':
                rate = tick_004796
            else:
                rate = tick_000540
            days = 0
            if Holidays.__contains__(AddDays(temp[20],2)):
                days = Holidays[AddDays(temp[20],2)]
            if temp[20].strftime('%w') == '4' and not Holidays.__contains__(AddDays(temp[20],1)) and not Holidays.__contains__(AddDays(temp[20],2)):
                days = 3
            index = np.where(dt == rate)
            if len(index[0]) > 0:
                cc_lirun = 0
                index = index[0][0]
                qty = 0
                if temp[12] == 0:
                    qty = temp[9] * 100
                else:
                    qty = temp[12]
                cc_lirun = float(rate[index][3]) * qty / 1000000
                if days > 0:
                    for k in range(1,days):
                        if len(rate) > index + k:
                            cc_lirun += float(rate[index+k][3]) * qty / 1000000
                cc += cc_lirun
                total += cc_lirun
        elif temp[11] != 0 and temp[18] != 0:
            if temp[22] == '':
                jy_lirun = temp[18] * temp[19] - temp[11] * temp[12]
                jy += jy_lirun
                total += float(jy_lirun)
            elif temp[22] == 'guoyedone':
                dt = temp[20].strftime('%Y-%m-%d')
                index = np.where(dt == tick_rate)
                if len(index[0]) > 0:
                    cc_lirun = 0
                    index = index[0][0]
                    if temp[2] == '511990':
                        cc_lirun = float(tick_rate[index][3]) * temp[12] / 100 + float(temp[18]) * temp[19] - float(temp[11]) * temp[12]
                        if temp[20].strftime('%w') == '5' or Holidays.__contains__(AddDays(temp[20],1)):
                            if len(tick_rate) > index + 1:
                                cc_lirun += float(tick_rate[index+1][3]) * temp[12] / 100
                        cc += cc_lirun
                        total += cc_lirun
                    elif temp[2] == '511660':
                        cc_lirun = float(tick_rate[index][3]) * temp[12] / 100 + float(temp[18]) * temp[19] - float(temp[11]) * temp[12]
                        days = 0
                        if Holidays.__contains__(AddDays(temp[20],1)):
                            days = Holidays[AddDays(temp[20],1)]
                        elif temp[20].strftime('%w') == '5':
                            days = 3
                        if days > 0:
                            for k in range(1,days):
                                cc_lirun += float(tick_rate[index+k][3]) * temp[12] / 100
                        cc += cc_lirun
                        total += cc_lirun
                if temp[2] == '511880':
                    cc_lirun = float(temp[18]) * temp[19] - float(temp[11]) * temp[12]
                    cc += cc_lirun
                    total += cc_lirun
    gain.append(total)
    jiaoyi.append(jy)
    chicangguoye.append(cc)
    nihuigou.append(nhg)
    gain.append(sum(gain))
    policy_gain.append(gain)
    jiaoyi.append(sum(jiaoyi))
    policy_jiaoyi.append(jiaoyi)
    chicangguoye.append(sum(chicangguoye))
    policy_chicang.append(chicangguoye)
    nihuigou.append(sum(nihuigou))
    policy_nihuigou.append(nihuigou)
    dates.append('总计')
    policy_date.append(dates)

    array = np.zeros([len(policy_gain[policy_index]),6],dtype=object)
    total_day = 0
    for j in range(len(policy_gain[policy_index])):
        array[j][0] = per_capital[policy_index]
        array[j][1] = math.floor(policy_jiaoyi[policy_index][j] * 100) / 100
        array[j][2] = math.floor(policy_nihuigou[policy_index][j] * 100) / 100
        array[j][3] = math.floor(policy_chicang[policy_index][j] * 100) / 100
        array[j][4] = math.floor(policy_gain[policy_index][j] * 100) / 100
        if j != len(policy_gain[policy_index]) - 1:
            date = datetime.datetime.strptime(policy_date[policy_index][j],'%Y-%m-%d')
            if Holidays.__contains__(AddDays(date,1)):
                days = Holidays[AddDays(date,1)]
            else:
                if date.strftime('%w') == '5':
                    days = 3
                else:
                    days = 1
            total_day += days
            array[j][5] = round(policy_gain[policy_index][j] / array[j][0] / days * 36500,2)
        else:
            array[j][5] = round(policy_gain[policy_index][j] / array[j][0] / total_day * 36500,2)

        array[j][5] = str(array[j][5]) + '%'
    df = pd.DataFrame(array,index=policy_date[policy_index],columns=['本金','日内交易收益','逆回购收益','持仓过夜收益','当日总收益','当日折合年化收益率'])
    df.to_excel(writer,sheet_name=policy[6:])
# df = pd.DataFrame(array,index=policy_date[policy_index],columns=['本金','日内交易收益','逆回购收益','持仓过夜收益','当日总收益','当日年化收益'])
# df.to_excel(writer,sheet_name=policy[6:])
array = np.zeros([len(max(policy_date))-1,5],dtype=object)
leijishouyilv = 0
days = 0
for i,date in enumerate(max(policy_date)[:-1]):
    benjin = 0
    shouyi = 0
    for j,temp_date in enumerate(policy_date):
        if date in temp_date:
            index = np.where(date == np.array(temp_date))[0][0]
            benjin += per_capital[j]
            shouyi += policy_gain[j][index]
    date = datetime.datetime.strptime(date,'%Y-%m-%d')
    if Holidays.__contains__(AddDays(date,1)):
        day = Holidays[AddDays(date,1)]
    else:
        if date.strftime('%w') == '5':
            day = 3
        else:
            day = 1
    days += day
    shouyi = math.floor(shouyi * 100) / 100
    array[i][0] = benjin
    array[i][1] = shouyi
    array[i][2] = str(round(shouyi / benjin / day * 36500,2)) + '%'
    leijishouyilv += shouyi / benjin * 36500
    array[i][3] = str(round(leijishouyilv / (days),2)) + "% (累计" + str(days) + "天）"
    jingzhi = round((shouyi / benjin + 1) * jingzhi,7)
    array[i][4] = jingzhi
df = pd.DataFrame(array,index=max(policy_date)[:-1],columns=['本金','当日总收益','当日折合年化收益率','累计折合年化收益率','净值'])
df.to_excel(writer,sheet_name='统计')
writer.save()

excels = file_name(account)
writer2 = pd.ExcelWriter(account + '/统计/统计.xlsx')
data = np.array([])
dates = np.array([])
for excel in excels:
    df = pd.read_excel(excel,'统计')
    if (len(data) > 0):
        data = np.append(data,np.array(df),axis=0)
        dates = np.append(dates,np.array(df.index),axis=0)
    else:
        data = np.array(df)
        dates = np.array(df.index)
array = np.zeros([len(dates),6],dtype=object)
sum_money = 0
days = 0
for i in range(len(data)):
    date = datetime.datetime.strptime(dates[i],'%Y-%m-%d')
    if Holidays.__contains__(AddDays(date,1)):
        day = Holidays[AddDays(date,1)]
    else:
        if date.strftime('%w') == '5':
            day = 3
        else:
            day = 1
    days += day
    sum_money += data[i][1]
    array[i][0] = data[i][0]
    array[i][1] = data[i][1]
    array[i][2] = sum_money
    array[i][3] = str(round(array[i][1] / array[i][0]  * 36500,2)) + '%'
    array[i][4] = str(round(sum_money / array[i][0] * 36500 / days,2)) + '% (累计'+ str(days) + '天）'
    array[i][5] = data[i][4]
df = pd.DataFrame(array,index=dates,columns=['本金','当日总收益','累计收益','当日折合年化收益率','累计折合年化收益率','净值'])
df.to_excel(writer2,sheet_name='统计')
writer2.save()
# plt.figure('capital')
# for index in range(len(policy_gain)):
#     print(policy_gain[index])
#     plt.plot(policy_date[index],policy_gain[index])
# plt.grid(True)
# plt.gcf().autofmt_xdate()  # 自动旋转日期标记
# plt.show()
# writer = pd.ExcelWriter(account + '/货基收益明细%' + account + '%' + trade_time + '.xlsx')
