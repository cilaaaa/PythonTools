import time
import json
import pymssql
import threading
import websocket
import traceback
from time import sleep
import datetime


data = {}
trade_data = []
symbols = ["orderBook10:XBTUSD","trade:XBTUSD"]
pre_save_data = []
data_count = 0
pre_save_trade_data = []
trade_data_count = 0
def __on_message(ws, message):
    '''Handler for parsing WS messages.'''
    global data_count,trade_data_count
    message = json.loads(message)
    table = message['table'] if 'table' in message else None
    action = message['action'] if 'action' in message else None
    symbol = message['data'][0]['symbol'] if 'data' in message else None
    try:
        if 'subscribe' in message:
            print("Subscribed to %s." % message['subscribe'])
        elif action and table == 'orderBook10':
            if symbol not in data:
                data[symbol] = {}
                data[symbol]['ask'] = {}
                data[symbol]['asks'] = {}
                data[symbol]['bid'] = {}
                data[symbol]['bids'] = {}

            # There are four possible actions from the WS:
            # 'partial' - full table image
            # 'insert'  - new row
            # 'update'  - update row
            # 'delete'  - delete row
            orderBooks = message['data'][0]
            for i in range(10):
                data[symbol]['ask'][i] = orderBooks['asks'][i][0]
                data[symbol]['asks'][i] = orderBooks['asks'][i][1]
                data[symbol]['bid'][i] = orderBooks['bids'][i][0]
                data[symbol]['bids'][i] = orderBooks['bids'][i][1]
            merge_data = {}
            merge_data['timestamp'] =  int(round(time.time() * 1000))
            merge_data['data'] = data
            merge_data['symbol'] = symbol
            pre_save_data.append(merge_data)
            if len(pre_save_data) > 100:
                receive_tick(pre_save_data)
                data_count += 100
                print(data_count)
                pre_save_data.clear()
        elif action and table == 'trade':
            trades = message['data']
            if action == 'partial':
                for trade in trades:
                    trade_data.append(trade)
                    pre_save_trade_data.append(trade)
                # Keys are communicated on partials to let you know how to uniquely identify
                # an item. We use it for updates.
            elif action == 'insert':
                for trade in trades:
                    trade_data.append(trade)
                    pre_save_trade_data.append(trade)
            if len(pre_save_trade_data) > 100:
                receive_trade(pre_save_trade_data)
                trade_data_count += 100
                print(trade_data_count)
                pre_save_trade_data.clear()
    except:
        print(traceback.format_exc())

def __on_error(ws, error):
    print(traceback.format_exc())
    print("WB Error : %s" % error)

def __on_open(ws):
    '''Called when the WS opens.'''
    print("Websocket Opened.")
    __send_command(default_ws,"subscribe",symbols)

def __on_close(ws):
    '''Called on websocket close.'''
    print('Websocket Closed')
    ws.run_forever()
    # startWs(ws)

def __send_command(ws,command, args=None):
    '''Send a raw command.'''
    if args is None:
        args = []
    msg = json.dumps({"op": command, "args": args})
    print(msg)
    ws.send(msg)

def exit(ws):
    '''Call this to exit - will close websocket.'''
    ws.close()
    startWs(ws)



def startWs(ws):
    wst = threading.Thread(target=lambda: ws.run_forever())
    wst.daemon = True
    wst.start()
    print("Started thread")

url = 'wss://www.bitmex.com/realtime'
default_ws = websocket.WebSocketApp(url,
                             on_message=__on_message,
                             on_close=__on_close,
                             on_open=__on_open,
                             on_error=__on_error)

def receive_trade(trade_data):
    '''
    接收tick数据并存储到数据库
    :param timeStamp: 作为数据库表key值
    :param json_2_dict: tick数据
    :return:
    '''
    sql = 'INSERT INTO tick_data_bitmex_trade(tick_code,tick_time,timestamp,price,size,side) VALUES '
    for trade in trade_data:
        symbol = trade['symbol']
        utc = trade['timestamp']
        price = trade['price']
        size = trade['size']
        side = trade['side']
        UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
        utcTime = datetime.datetime.strptime(utc, UTC_FORMAT)
        localtime = utcTime + datetime.timedelta(hours=8)
        timestamp = int(localtime.timestamp() * 1000)
        if len(str(localtime)) == 26:
            trade_time = str(localtime)[:-3]
        else:
            trade_time = str(localtime)

        sql += "('%s', '%s', '%s', '%s', '%s', '%s')," % (symbol,trade_time,timestamp,price,size,side)
    sql = sql[:-1]
    con = pymssql.connect('127.0.0.1', 'sa', 'sa123!@#', 'Bitcoin')
    # con = mysql.connector.connect(host='localhost', port=3306, user='root',
    #                               password='1234', database='huobi', charset='utf8')

    cursor = con.cursor()
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        con.commit()

        # logger.info('数据保存成功')
    except pymssql.Error as e:
        print(' sql:' + sql + ' insert error!{}'.format(e))
        # 如果发生错误则回滚
        con.rollback()

        cursor.close()
        # 关闭数据库连接
        con.close()

def receive_tick(merge_data):
    '''
    接收tick数据并存储到数据库
    :param timeStamp: 作为数据库表key值
    :param json_2_dict: tick数据
    :return:
    '''
    sql = 'INSERT INTO tick_data_bitmex(TICK_CODE,tick_timestamp,TICK_TIME, TICK_ASK1, TICK_ASKS1,TICK_ASK2, TICK_ASKS2,TICK_ASK3, TICK_ASKS3,TICK_ASK4, TICK_ASKS4,TICK_ASK5, TICK_ASKS5,TICK_ASK6, TICK_ASKS6,' \
          'TICK_ASK7, TICK_ASKS7,TICK_ASK8, TICK_ASKS8,TICK_ASK9, TICK_ASKS9,TICK_ASK10, TICK_ASKS10,' \
          'TICK_BID1,TICK_BIDS1,TICK_BID2,TICK_BIDS2,TICK_BID3,TICK_BIDS3,TICK_BID4,TICK_BIDS4,TICK_BID5,TICK_BIDS5,TICK_BID6,TICK_BIDS6,TICK_BID7,TICK_BIDS7,TICK_BID8,TICK_BIDS8,TICK_BID9,TICK_BIDS9,TICK_BID10,TICK_BIDS10) ' \
          'VALUES '
    for data in merge_data:
        tick = data['data']
        timestamp = data['timestamp']
        tick_time = datetime.datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')
        if len(tick_time) == 26:
            tick_time = tick_time[:-3]
        TICK_CODE = data['symbol']
        sql += "('%s', '%s', '%s'," % (TICK_CODE, timestamp, tick_time)
        for i in range(10):
            sql += "'" + str(tick[TICK_CODE]['ask'][i]) + "','" + str(tick[TICK_CODE]['asks'][i]) + "',"
        for i in range(10):
            sql += "'" + str(tick[TICK_CODE]['bid'][i]) + "','" + str(tick[TICK_CODE]['bids'][i]) + "',"
        sql = sql[:-1] + "),"
    sql = sql[:-1]
    con = pymssql.connect('127.0.0.1', 'sa', 'sa123!@#', 'Bitcoin')
    # con = mysql.connector.connect(host='localhost', port=3306, user='root',
    #                               password='1234', database='huobi', charset='utf8')

    cursor = con.cursor()
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        con.commit()

        # logger.info('数据保存成功')
    except pymssql.Error as e:
        print(' sql:' + sql + ' insert error!{}'.format(e))
        # 如果发生错误则回滚
        con.rollback()

        cursor.close()
        # 关闭数据库连接
        con.close()



def Call_Restful():
    startWs(default_ws)
    # Wait for connect before continuing
    while True:
        conn_timeout = 5
        while not default_ws.sock or not default_ws.sock.connected and conn_timeout:
            print(conn_timeout)
            sleep(1)
            conn_timeout -= 1
            if conn_timeout < 0:
                conn_timeout = 5
                exit(default_ws)
        sleep(1)



def market_depth():
    '''Get market depth (orderbook). Returns all levels.'''
    return data['orderBookL2']

def main():
    print('打开日志记录')

    Call_Restful()

    #list = ['BTCUSDT', 'ETHUSDT', 'ETHBTC', 'EOSBTC', 'ONTBTC', 'EOSETH', 'ONTETH', "XRPUSDT", "DTAUSDT", "ZILUSDT", "LETUSDT", "ADAUSDT","EOSUSDT","QKCETH","QKCBTC",""]
    #list = ["XBTUSD", "XBTU18", "XBTZ18", "EOSU18", 'ETHUSD', "ETHU18"]
    # list = ["XBTUSD"]
    # q = []
    # for i in range(len(list)):
    #     q.append(threading.Thread(target=Call_Restful, name='BITMEX_Thread_' + list[i], args=(list[i],)))
    # print(q)
    # for t in q:
    #     t.start()
    # for t in q:
    #     t.join()


if __name__ == '__main__':
    main()