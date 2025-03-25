"""
预测各国可再生能源占比的未来趋势
使用随机森林回归模型，结合历史数据和社会经济指标
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path):
    """加载并预处理数据"""
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 选择相关特征
    selected_features = [
        # 基础标识
        'country', 'year', 'iso_code',
        
        # 经济人口
        'gdp', 'population',
        
        # 能源结构
        'renewables_share_energy',          
        'fossil_share_energy',
        'low_carbon_share_energy',
        'coal_share_energy',
        'oil_share_energy',
        
        # 环境效应
        'carbon_intensity_elec',
        'greenhouse_gas_emissions',
        
        # 电力特征
        'electricity_demand_per_capita',
        'renewables_elec_per_capita',
        'solar_share_elec',
        'wind_share_elec',
        'nuclear_share_elec',
        
        # 经济效率
        'energy_per_gdp'
    ]
    
    # 只选择存在的列
    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features].copy()
    
    # 将无穷大值替换为NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 输出初始数据
    df.to_csv('1_data_initial.csv', index=False)
    print("初始数据已保存到 1_data_initial.csv")
    
    # 添加时序特征
    for i in range(1, 6):
        df[f'renewable_share_lag_{i}'] = df.groupby('country')['renewables_share_energy'].shift(i)
    
    # 计算化石能源消费变化率
    df['fossil_cons_change_pct'] = df.groupby('country')['fossil_share_energy'].transform(
        lambda x: x.pct_change(fill_method=None)
    )
    
    # 添加政策相关特征
    df['paris_agreement'] = (df['year'] >= 2015).astype(int)
    
    # 安全地计算年度变化率
    df['renewable_growth'] = df.groupby('country')['renewables_share_energy'].transform(
        lambda x: x.pct_change(fill_method=None)
    )
    
    # 使用中位数填充NaN值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # 确保数据按国家和年份排序
    df = df.sort_values(['country', 'year'])
    
    # 输出特征工程后的数据
    df.to_csv('2_data_after_feature_engineering.csv', index=False)
    print("特征工程后的数据已保存到 2_data_after_feature_engineering.csv")
    
    return df

def prepare_features(df):
    """特征工程"""
    # 标准化数值特征前检查无穷大值
    numeric_features = [
        'gdp', 'population', 
        'carbon_intensity_elec', 'greenhouse_gas_emissions',
        'electricity_demand_per_capita', 'renewables_elec_per_capita',
        'energy_per_gdp',
        'fossil_cons_change_pct'
    ]
    
    # 只选择存在的特征
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    # 将极端值限制在合理范围内（使用分位数）
    for feature in numeric_features:
        q1 = df[feature].quantile(0.01)
        q3 = df[feature].quantile(0.99)
        df[feature] = df[feature].clip(q1, q3)
    
    # 输出处理极端值后的数据
    df.to_csv('3_data_after_outlier_handling.csv', index=False)
    print("处理极端值后的数据已保存到 3_data_after_outlier_handling.csv")
    
    # 标准化
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # 输出标准化后的数据
    df.to_csv('4_data_after_scaling.csv', index=False)
    print("标准化后的数据已保存到 4_data_after_scaling.csv")
    
    return df

def train_model(df, target_years=5):
    """训练随机森林模型"""
    # 准备特征，排除非数值列
    feature_columns = [col for col in df.columns 
                      if col not in ['country', 'year', 'renewables_share_energy', 'iso_code']]
    
    # 确保所有特征都是数值类型
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
    X = df[numeric_features]
    y = df['renewables_share_energy']
    
    # 确保没有无穷大值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # 输出最终用于训练的数据
    X.to_csv('5_data_final_for_training.csv', index=False)
    print("最终用于训练的数据已保存到 5_data_final_for_training.csv")
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 输出训练集和测试集
    X_train.to_csv('6_data_train.csv', index=False)
    X_test.to_csv('7_data_test.csv', index=False)
    print("训练集和测试集已分别保存到 6_data_train.csv 和 7_data_test.csv")
    
    # 训练模型（添加参数控制）
    model = RandomForestRegressor(
        n_estimators=100,  # 减少树的数量以提高稳定性
        max_depth=10,      # 限制树的深度
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 安全地训练模型
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print("训练时出错：", e)
        print("正在检查特征值范围...")
        for col in X_train.columns:
            print(f"{col}: 范围 [{X_train[col].min()}, {X_train[col].max()}]")
        raise
    
    return model, X_train, X_test, y_train, y_test

def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': model.feature_importances_
    })
    return importance.sort_values('重要性', ascending=False)

def predict_future(model, df, country, years_to_predict=5):
    """预测特定国家的未来趋势"""
    latest_data = df[df['country'] == country].iloc[-1:].copy()
    predictions = []
    
    # 获取训练模型时使用的特征（数值类型特征）
    feature_columns = [col for col in df.columns 
                      if col not in ['country', 'year', 'renewables_share_energy', 'iso_code']]
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
    
    for i in range(years_to_predict):
        # 更新年份
        latest_data['year'] = latest_data['year'] + 1
        
        # 更新滞后特征
        for j in range(5, 0, -1):
            if j == 1:
                latest_data[f'renewable_share_lag_{j}'] = predictions[-1] if predictions else latest_data['renewables_share_energy'].values[0]
            else:
                latest_data[f'renewable_share_lag_{j}'] = latest_data[f'renewable_share_lag_{j-1}']
        
        # 预测（使用与训练时相同的特征集）
        pred = model.predict(latest_data[numeric_features])
        predictions.append(pred[0])
    
    return predictions

def plot_predictions(country, historical_data, predictions, future_years):
    """绘制预测结果图表"""
    plt.figure(figsize=(12, 6))
    
    # 绘制历史数据
    plt.plot(historical_data['year'], historical_data['renewables_share_energy'], 
             label='历史数据', marker='o')
    
    # 绘制预测数据
    future_years = range(historical_data['year'].max() + 1, 
                        historical_data['year'].max() + len(predictions) + 1)
    plt.plot(future_years, predictions, label='预测数据', marker='s', linestyle='--')
    
    plt.title(f'{country}可再生能源占比预测')
    plt.xlabel('年份')
    plt.ylabel('可再生能源占比 (%)')
    plt.legend()
    plt.grid(True)
    
    return plt

def main():
    # 加载数据
    df = load_and_prepare_data('owid-energy-data.csv')
    df = prepare_features(df)
    
    # 训练模型
    model, X_train, X_test, y_train, y_test = train_model(df)
    
    # 输出模型性能
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"训练集 R2 分数: {train_score:.3f}")
    print(f"测试集 R2 分数: {test_score:.3f}")
    
    # 分析特征重要性
    importance = analyze_feature_importance(model, X_train.columns)
    print("\n特征重要性:")
    print(importance)
    
    # 预测示例（中国）
    country = 'China'
    predictions = predict_future(model, df, country, years_to_predict=10)
    historical_data = df[df['country'] == country]
    
    # 输出预测结果
    print(f"\n{country}未来10年可再生能源占比预测:")
    for year, pred in enumerate(predictions, start=historical_data['year'].max() + 1):
        print(f"{year}年: {pred:.2f}%")
    
    # 绘制预测图表
    plt = plot_predictions(country, historical_data, predictions, range(len(predictions)))
    plt.show()

if __name__ == "__main__":
    main()