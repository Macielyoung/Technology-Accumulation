## 常用的几种指令使用样例

1. 读取文件内容，同时匹配出需要查询的关键字（可多个）

   ```
   cat a.txt | grep -E "(a|b|c)" | head
   ```

2. 读取文件内容，过滤前几行不读

   ```
   cat a.txt | tail -n +3 | head
   ```

3. 将文件内容每两行合并为一行

   ```
   awk '{if(NR%2!=0)ORS=" ";else ORS="\n"}1' a.txt > b.txt
   xargs -L2 < a.txt > b.txt
   ```

4. 统计文件行数/字节数/字数

   ```
   wc -l a.txt
   wc -c a.txt
   wc -w a.txt
   ```

5. 统计指定项的频数（-F 用来分割行内容）

   ```
   awk -F" " '{a[$1";"$2]+=$3} END{for(b in a){print b";"a[b]}}' a.txt > b.txt
   ```

6. 