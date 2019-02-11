# coding:utf-8

# @Created : Macielyoung
# @Time : 2019/2/2
# @Function : generate QR code

import qrcode

url = "https://github.com/Macielyoung"
img = qrcode.make(url)
img.show()