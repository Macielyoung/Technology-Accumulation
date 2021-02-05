## Mongo CRUD操作方式

1. python访问mysql数据库

   ```python
   我们主要使用python编程语言，通过pymysql包来访问mongo数据库
   pip install pymysql
   ```

   

2. pymongo连接数据库

   ```python
   连接数据库有两种方式
   import pymysql
   1. conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db)
   2. config = {'host': host, 'port': port, 'user': user, 'password': password, 'db': db}
      conn = pymysql.connect(**config)
   
   创建游标
   cursor = conn.cursor()
   ```
   
   
   
3. Insert(插入数据)

   ```python
   1. 插入单条数据
   insert_one_sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
            						LAST_NAME, AGE, SEX, INCOME)
            						VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
   try:
       cursor.execute(insert_one_sql)
       # 提交数据库执行
       conn.commit()
   except:
     	conn.rollback()
   
     
   2. 插入多条数据
   insert_many_sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
            						 LAST_NAME, AGE, SEX, INCOME)
            						 VALUES ('Mac', 'Mohan', 20, 'M', 2000),
            						 				('Win', 'Nancy', 25, 'F', 1800)"""
   try:
       cursor.execute(insert_many_sql)
       # 提交数据库执行
       conn.commit()
   except:
     	conn.rollback()
   
   也可以使用executemany来执行
   try:
       cursor.executemany('''insert into app_notification(updated_datetime, created_datetime, title, title_en,
                                         content, content_en, notification_type, is_read, recipient_id)
                             values(%s, %s, %s, %s, %s, %s, %s, 0, %s)''', values_list)
       conn.commit()
   except:
     	conn.rollback()
   ```

   

4. Read(读取数据)

   ```python
   query_sql = "select * from employee"
   cursor.execute(query_sql)
   
   1. 获取第一行数据
   row1 = cursor.fetchone()
   
   2. 获取前n行数据
   rown = cursor.fetchmany(n)
   
   3. 获取所有数据
   rows = cursor.fetchall()
   
   conn.commit()
   ```

   

5. Update(更新数据)

   ```python
   try:
       update_sql = "update employee set INCOME = 2500 where LAST_NAME = 'Mohan'"
       curosr.execute(update_sql)
       conn.commit()
   except:
     	conn.rollback()
   ```
   
   
   
6. Delete(删除数据)

   ```python
   try:
       delete_sql = "delete from employee where LAST_NAME = 'Mohan'"
       cursor.execute(delete_sql)
       conn.commit()
   except:
     	conn.rollback()
   ```

   

7. 模糊查询

   ```python
   1. 通过like字段来匹配
   SELECT 字段 FROM 表 WHERE 某字段 Like 条件
   
   （1） %表示任意个字符，可匹配任意类型和长度的字符串
   （2） _表示任意单个字符，它常用来限制表达式的字符长度语句
   
   2. 通过regexp正则匹配
   
3. in查询
   
   4. like contact，和like比较类似
   ```
   
   

