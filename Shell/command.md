## 常用的几种指令使用样例

1. 读取文件内容，同时匹配出需要查询的关键字（可多个）

   ```shell
   cat a.txt | grep -E "(a|b|c)" | head
   ```

2. 读取文件内容，过滤前几行不读

   ```shell
   cat a.txt | tail -n +3 | head
   ```

3. 将文件内容每两行合并为一行

   ```shell
   awk '{if(NR%2!=0)ORS=" ";else ORS="\n"}1' a.txt > b.txt
   xargs -L2 < a.txt > b.txt
   ```

4. 统计文件行数/字节数/字数

   ```shell
   wc -l a.txt
   wc -c a.txt
   wc -w a.txt
   ```

5. 统计指定项的频数（-F 用来分割行内容）

   ```shell
   awk -F" " '{a[$1";"$2]+=$3} END{for(b in a){print b";"a[b]}}' a.txt > b.txt
   ```

6. 匹配某条记录同时返回之后的三行记录

   ```shell
   grep -A 3 "match" a.log > b.log
   ```

7. 根据grep后分割的多行进行合并，紧跟第6题（其中每四行中间有一个“—”行）

   ```shell
   awk '{if($0!="--"){a=a$0} else{print a; a=""}}' b.log > c.log
   ```

8. 使用多个分割符来对行数据进行处理

   ```shell
   awk -F"[-:,]" '{print $0}' a.log | head
   ```

9. 降序排序输出结果

   ```shell
   sort -n -r -k 3 -t ';' a.log
   ```

10. 句子频率统计并降序输出

  ```shell
  awk -F"[:,]" '{a[$2]+=1} END{for(b in a){print b" : "a[b]}}' a.log | sort -n -r -k 2 -t ":" | head -n 50
  ```

11. 针对shell变量进行替换操作(针对x变量将所有的‘|’符号换成空格符，其中‘//’表示替换所有)

    ```shell
    ${x//\|/\ }
    ```

12. 将内容写入文件和追加入文件

    ```shell
    echo "$x" > a.log
    echo "$x" >> a.log
    ```

13. 全局替换文件中的内容，比如把t-shirts替换为Tshirts。

    ```shell
    sed 's/t-shirts/Tshirts/g' < a.log > b.log
    ```

14. 截断每行字符串

    ```shell
    # 保留第10个字符往后
    cut -c 10- a.log > b.log
    # 以":"分割字符，输出第3-5列
    cut -d ":" -f 3-5
    ```

15. awk分割字符串后再匹配输出

    ```shell
    # 以";"分割字符串后匹配第一列中含有bananas的行
    awk -F ";" '$1 ~ /bananas/ {print $0}' a.log | head
    ```

16. 
