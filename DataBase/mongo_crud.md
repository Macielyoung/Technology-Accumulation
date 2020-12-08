## Mongo CRUD操作方式

1. python访问mongo数据库

   ```python
   我们主要使用python编程语言，通过pymongo包来访问mongo数据库
   pip install pymongo
   ```

   

2. pymongo连接数据库

   ```python
   连接数据库有两种方式
   from pymongo import MongoClient
   1. client = MongoClient(host, port)
   2. client = MongoClient(uri)
   第一种方式比较常用。
   
   选择对应的数据库
   1. db = client[db_name]
   2. db = client.db_name
   
   选择对应的表（如果不存在则新建）
   1. connection = db[collect_name]
   2. connection = db.collect_name
   ```

   

3. Create / Insert(新建或插入数据)

   ```python
   1. insert_one() 插入单条数据
   connection.insert_one(
     {
       "name": 'maciel',
       "age": 26,
     }
   )
   
   2. insert_many() 插入多条数据
   connection.insert_many(
     [
       {"name": 'maciel', "age": 26},
       {"name": 'jordan', "age": 51},
       {"name": 'steve': "age": 55},
     ]
   )
   
   3. insert() 插入单条或多条数据
   connection.insert({"name": 'maciel', "age": 26})
   connection.insert([{"name": 'maciel', "age": 26}, {"name": 'jordan', "age": 51}, {"name": 'steve': "age": 55},])
   ```

   

4. Read(读取数据)

   ```python
   1. find() 返回表中所有数据
   connection.find()
   
   2. find_one() 返回表中第一条数据
   connection.find_one()
   
   3. 上面两种方法都可以加上条件查询数据
   connection.find({"name": "maciel"})
   connection.find_one({"name": "maciel"})
   ```

   

5. Update(更新数据)

   ```python
   1. update_one() 更新一条记录
   connection.update_one({"name": "maciel"}, {"$set": {"name": "mac"}})
   
   2. update_many() 更新多条记录
   connection.update_many({"name": "maciel"}, {"$set": {"name": "mac"}})
   
   3. update()
   connection.update({"name": "maciel"}, {"$set": {"name": "mac"}})
   
   4. replace_one()
   connection.replace_one({"name": "maciel"}, {"$set": {"name": "mac"}})
   
   这四条语法都满足以下的语法：
   <method_name>(condition, update_or_replace_document, upsert=False, bypass_document_validation=False)
   
   condition: A query that matches the document to replace.
   update_or_replace_document: The new document.
   upsert (optional): If True, perform an insert if no documents match the filter.
   bypass_document_validation: (optional) If True, allows the write to opt-out of document level validation. Default is False.
   ```

   

6. Delete(删除数据)

   ```python
   1. delete_one()
   connection.delete_one({"name": "maciel"})
   
   2. delete_many()
   connection.delete_many("name": "maciel")
   ```

   

7. 模糊查询

   ```python
   mongo支持模糊匹配，使用正则表达式来实现。
   1. 姓名中包含mac字段
   connection.find({"name": {$regex: "mac"}})
   connection.find({"name": /mac/})
   
   2. 姓名中包含mac字段，不区分大小写
   connection.find({"name": {$regex: "mac", $options: "$i"}})
   ```

   

