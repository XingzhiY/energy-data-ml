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

def fill_missing_by_country(df, columns):
    """
    按国家分别填充缺失值
    
    Args:
        df: 数据框
        columns: 需要填充的列名列表
    
    Returns:
        填充后的数据框
    """
    df_filled = df.copy()
    
    # 对每个国家分别处理
    for country in df['country'].unique():
        country_mask = df['country'] == country
        
        # 对每个数值列进行处理
        for col in columns:
            # 获取该国家该列的数据
            country_data = df.loc[country_mask, col]
            
            # 如果该国家有非空值，使用该国家的中位数填充
            if not country_data.isna().all():
                median_value = country_data.median()
                df_filled.loc[country_mask, col] = country_data.fillna(median_value)
            else:
                # 如果该国家所有值都是空的，使用全球中位数填充
                global_median = df[col].median()
                df_filled.loc[country_mask, col] = global_median
    
    return df_filled

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
    
    # 获取所有数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 按国家分别填充缺失值
    df = fill_missing_by_country(df, numeric_columns)
    
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

def train_model(df, test_years=5):
    """
    训练随机森林模型，使用时间序列划分方法
    
    Args:
        df: 数据框
        test_years: 用作测试的年份数量
    """
    # 准备特征，排除非数值列
    feature_columns = [col for col in df.columns 
                      if col not in ['country', 'year', 'renewables_share_energy', 'iso_code']]
    
    # 确保所有特征都是数值类型
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
    
    # 按时间划分训练集和测试集
    cutoff_year = df['year'].max() - test_years
    
    # 训练集：所有早于cutoff_year的数据
    train_mask = df['year'] <= cutoff_year
    # 测试集：所有晚于cutoff_year的数据
    test_mask = df['year'] > cutoff_year
    
    # 准备特征矩阵
    X = df[numeric_features]
    y = df['renewables_share_energy']
    
    # 确保没有无穷大值
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 按国家分别填充缺失值
    X = fill_missing_by_country(pd.concat([df[['country']], X], axis=1), numeric_features)[numeric_features]
    
    # 划分训练集和测试集
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # 输出训练集和测试集的时间范围
    print(f"训练集时间范围：{df[train_mask]['year'].min()} - {df[train_mask]['year'].max()}")
    print(f"测试集时间范围：{df[test_mask]['year'].min()} - {df[test_mask]['year'].max()}")
    
    # 输出最终用于训练的数据
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv('6_data_train_time_series.csv', index=False)
    test_df.to_csv('7_data_test_time_series.csv', index=False)
    print("最终用于训练的数据已保存到 6_data_train_time_series.csv 和 7_data_test_time_series.csv")
    
    # 训练模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
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
    
    # 计算每个国家的预测性能
    countries = df['country'].unique()
    country_scores = {}
    
    for country in countries:
        country_mask_test = (df['country'] == country) & test_mask
        if sum(country_mask_test) > 0:  # 确保该国家在测试集中有数据
            X_test_country = X[country_mask_test]
            y_test_country = y[country_mask_test]
            score = model.score(X_test_country, y_test_country)
            country_scores[country] = score
    
    # 输出每个国家的预测性能
    print("\n各国预测性能 (R² 分数):")
    for country, score in sorted(country_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{country}: {score:.3f}")
    
    return model, X_train, X_test, y_train, y_test, country_scores

def evaluate_with_time_series_cv(df, n_splits=5):
    """
    使用时间序列交叉验证评估模型
    
    Args:
        df: 数据框
        n_splits: 交叉验证折数
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    # 准备特征
    feature_columns = [col for col in df.columns 
                      if col not in ['country', 'year', 'renewables_share_energy', 'iso_code']]
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
    
    X = df[numeric_features]
    y = df['renewables_share_energy']
    
    # 处理无穷大值和异常值
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 按国家分别填充缺失值
    X = fill_missing_by_country(pd.concat([df[['country']], X], axis=1), numeric_features)[numeric_features]
    
    # 对每一列分别处理异常值
    for col in X.columns:
        # 计算分位数
        q1 = X[col].quantile(0.01)
        q3 = X[col].quantile(0.99)
        # 将超出范围的值限制在分位数范围内
        X[col] = X[col].clip(q1, q3)
    
    # 确保所有值都是有限的
    assert np.all(np.isfinite(X)), "数据中仍然存在无穷大值"
    
    # 初始化时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 初始化模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 存储每次交叉验证的分数
    cv_scores = []
    
    # 进行交叉验证
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        try:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)
            print(f"Fold {fold} R² 分数: {score:.3f}")
        except Exception as e:
            print(f"Fold {fold} 训练失败: {str(e)}")
            continue
    
    if cv_scores:
        print(f"\n平均 R² 分数: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    else:
        print("所有折都训练失败")
    
    return cv_scores

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
    
    # 使用时间序列交叉验证评估模型
    print("\n执行时间序列交叉验证...")
    cv_scores = evaluate_with_time_series_cv(df)
    
    # 训练最终模型
    print("\n训练最终模型...")
    model, X_train, X_test, y_train, y_test, country_scores = train_model(df, test_years=5)
    
    # 输出整体模型性能
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\n整体模型性能:")
    print(f"训练集 R2 分数: {train_score:.3f}")
    print(f"测试集 R2 分数: {test_score:.3f}")
    
    # 分析特征重要性
    importance = analyze_feature_importance(model, X_train.columns)
    print("\n特征重要性:")
    print(importance)
    
    # 为几个主要国家进行预测
    main_countries = ['China', 'United States', 'India', 'Germany', 'Japan']
    for country in main_countries:
        if country in df['country'].unique():
            predictions = predict_future(model, df, country, years_to_predict=10)
            historical_data = df[df['country'] == country]
            
            print(f"\n{country}未来10年可再生能源占比预测:")
            for year, pred in enumerate(predictions, start=historical_data['year'].max() + 1):
                print(f"{year}年: {pred:.2f}%")
            
            # 绘制预测图表
            plt = plot_predictions(country, historical_data, predictions, range(len(predictions)))
            plt.savefig(f'prediction_{country}.png')
            plt.close()

if __name__ == "__main__":
    main()