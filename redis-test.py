# pip install redis  # 安装redis库

import redis

r = redis.Redis(host='121.48.161.96', port=6379, db=0)


# 增：添加数据
def add_data(key, value):
    r.set(key, value)           # set：设置键值对
    print(f"添加数据: {key} -> {value}")

# 查：获取数据
def get_data(key):
    value = r.get(key)         # get：获取键值对
    if value is not None:
        print(f"获取数据: {key} -> {value.decode('utf-8')}")
    else:
        print(f"键 {key} 不存在")

# 改：更新数据
def update_data(key, new_value):
    if r.exists(key):           # exists：判断键是否存在
        r.set(key, new_value)   # set：设置键值对，如果键存在，则更新键值对
        print(f"更新数据: {key} -> {new_value}")
    else:
        print(f"键 {key} 不存在，无法更新")

# 删：删除数据
def delete_data(key):
    if r.delete(key):    # delete：删除键值对
        print(f"删除数据: {key}")
    else:
        print(f"键 {key} 不存在，无法删除")


# # 添加数据
add_data("name", "muyu")
add_data("age", "30")

# 查数据
get_data("name")
get_data("age")

delete_data("name")
delete_data("age")