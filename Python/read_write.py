# coding:utf-8

# @Created : Macielyoung
# @Time : 2019/1/16
# @Function : read/write txt/csv with Python
import csv

# 读取txt文件，encoding用于处理编码格式
def read_txt(path):
	res = []
	with open(path, 'r', encoding='utf-8') as reader:
		for line in reader.readlines():
			res.append(line.strip('\n'))
	return res

# 单行写入txt文件
def write_single_txt(contents):
	with open('morn.txt', 'w', encoding='utf-8') as reader:
		for each in contents:
			reader.write(each+'\n')

# 多行写入txt文件
def write_multiple_txt(contents):
	contents = [each+'\n' for each in contents]
	with open('morn.txt', 'w', encoding='utf-8') as writer:
		writer.writelines(contents)

# 读取csv文件
def read_csv(path):
	all_row = []
	with open(path) as f:
		fcsv = csv.reader(f)
		# headers = next(fcsv)
		# print(headers)
		for row in fcsv:
			all_row.append(row)
			print(row)
	return all_row

# 字典序列读取csv文件
def read_dict_csv(path):
	all_row = []
	with open(path) as f:
		fcsv = csv.DictReader(f)
		for row in fcsv:
			all_row.append(list(row.values()))
			print(list(row.values()))
	return all_row

# 写入csv文件
def write_csv(headers, contents):
	with open('stocks.csv', 'w') as f:
		fcsv = csv.DictWriter(f, headers)
		fcsv.writeheader()
		fcsv.writerows(contents)

if __name__ == '__main__':
	res = read_txt('morning.txt')
	print(res)

	write_single_txt(res)
	write_multiple_txt(res)

	read_csv('stock.csv')
	read_dict_csv('stock.csv')

	headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume']
	rows = [{'Symbol':'AA', 'Price':39.48, 'Date':'6/11/2007',
		'Time':'9:36am', 'Change':-0.18, 'Volume':181800},
		{'Symbol':'AIG', 'Price': 71.38, 'Date':'6/11/2007',
		'Time':'9:36am', 'Change':-0.15, 'Volume': 195500},
		{'Symbol':'AXP', 'Price': 62.58, 'Date':'6/11/2007',
		'Time':'9:36am', 'Change':-0.46, 'Volume': 935000},
		]
	write_csv(headers, rows)