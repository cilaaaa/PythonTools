__author__ = 'Cila'
import urllib
import re

def getHtmlData(url):
    # 请求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166  Safari/535.19'}
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request)
    data = response.read()
    # 设置解码方式
    data = data.decode('utf-8')
    return data
for i in range(1,999999):
    url = "http://hrcxi.cn.com/?TWI=2sNAA0Bt07tAhW3eta&page=VEI="
    try:
        data = getHtmlData(url)
        print(i)
    except:
        data = ""

    # title = re.findall("<h4 class=\"title\">(.*)</h4>",data)
    # if len(title)> 0:
    #     if("蛊魂铃" in title[0]):
    #         print(title[0] + ": " +url)