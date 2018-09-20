#-*- coding: UTF-8 -*-

# 折半查找
def findk(x, y, xBeg, xEnd, yBeg, yEnd, k):
    xMid = (xBeg+xEnd)//2
    yMid = (yBeg+yEnd)//2
    halfLen = xMid-xBeg+yMid-yBeg+2
    if x[xMid] < y[yMid]:
        if k < halfLen:
            yEnd = yMid - 1
        else:
            k -= (xMid - xBeg + 1)
            xBeg = xMid + 1
    else:
        if k < halfLen:
            xEnd=xMid-1
        else:
            k -= (yMid - yBeg + 1)
            yBeg = yMid + 1
    if xBeg > xEnd:
        return y[yBeg + k - 1]
    if yBeg > yEnd:
        return x[xBeg + k - 1]
        # return
    res = findk(x, y, xBeg, xEnd, yBeg, yEnd, k)
    return res

if __name__ == "__main__":
    m, n, k = map(int, raw_input().split())
    x = map(int, raw_input().split())
    y = map(int, raw_input().split())

    if(k==1):
        print(min(x[0], y[0]))
    elif(k==m+n):
        print(max(x[-1], y[-1]))
    else:
        res = findk(x, y, 0, m-1, 0, n-1, k)
        print(res)
