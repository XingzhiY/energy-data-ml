import pandas as pd

# 加载数据
df = pd.read_csv('owid-energy-data.csv')

# 显示所有列名
print("数据集中的所有列：")
print(df.columns.tolist())

# 显示数据的基本信息
print("\n数据集基本信息：")
print(df.info())