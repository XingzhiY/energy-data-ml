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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

def filter_early_missing_data(df, missing_threshold=0.3):
    """
    过滤掉每个国家早期数据严重缺失的年份
    
    Args:
        df: 数据框
        missing_threshold: 缺失值比例阈值，超过此阈值的年份将被过滤
    
    Returns:
        过滤后的数据框
    """
    # 获取所有数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 存储每个国家的起始年份
    country_start_years = {}
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country].copy()
        
        # 按年份计算缺失值比例
        missing_by_year = country_data[numeric_columns].isna().mean(axis=1)
        
        # 找到第一个连续5年缺失值比例都低于阈值的年份
        window_size = 5
        valid_years = []
        
        for i in range(len(country_data) - window_size + 1):
            window_missing = missing_by_year.iloc[i:i+window_size]
            if all(window_missing < missing_threshold):
                valid_years.append(country_data.iloc[i]['year'])
                break
        
        if valid_years:
            country_start_years[country] = valid_years[0]
        else:
            # 如果找不到连续5年都满足条件的年份，删除该国家的所有数据
            country_start_years[country] = None
    
    # 过滤数据
    filtered_data = []
    for country in df['country'].unique():
        if country_start_years[country] is not None:
            country_data = df[df['country'] == country].copy()
            start_year = country_start_years[country]
            filtered_country_data = country_data[country_data['year'] >= start_year]
            filtered_data.append(filtered_country_data)
    
    # 合并所有过滤后的数据
    df_filtered = pd.concat(filtered_data, ignore_index=True)
    
    # 输出过滤信息
    # print("\n数据过滤信息:")
    # for country in df['country'].unique():
    #     original_years = df[df['country'] == country]['year'].nunique()
    #     if country_start_years[country] is not None:
    #         filtered_years = df_filtered[df_filtered['country'] == country]['year'].nunique()
    #         start_year = country_start_years[country]
    #         print(f"{country}: 原始年份数 {original_years} -> 过滤后年份数 {filtered_years} (起始年份: {start_year})")
    #     else:
    #         print(f"{country}: 原始年份数 {original_years} -> 已删除 (未找到连续5年数据质量合格的起始年份)")
    
    return df_filtered

def analyze_data_quality(df):
    """
    分析数据质量并筛选出数据质量较好的国家
    
    Args:
        df: 数据框
    
    Returns:
        筛选后的数据框
    """
    # 获取所有数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 计算每个国家的数据质量指标
    country_quality = {}
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        
        # 计算缺失值比例
        missing_ratio = country_data[numeric_columns].isna().mean().mean()
        
        # 计算数据时间跨度
        year_span = country_data['year'].max() - country_data['year'].min()
        
        # 计算数据点数量
        data_points = len(country_data)
        
        # 计算关键指标的缺失情况
        key_indicators = ['renewables_share_energy', 'gdp', 'population', 'carbon_intensity_elec']
        key_missing_ratio = country_data[key_indicators].isna().mean().mean()
        
        country_quality[country] = {
            'missing_ratio': missing_ratio,
            'year_span': year_span,
            'data_points': data_points,
            'key_missing_ratio': key_missing_ratio
        }
    
    # 转换为DataFrame
    quality_df = pd.DataFrame.from_dict(country_quality, orient='index')
    
    # 设置筛选条件
    quality_thresholds = {
        'missing_ratio': 0.2,  # 总体缺失值比例不超过20%（因为已经过滤了早期数据）
        'year_span': 10,       # 至少10年的数据
        'data_points': 10,     # 至少10个数据点
        'key_missing_ratio': 0.15  # 关键指标缺失值比例不超过15%
    }
    
    # 筛选出满足条件的国家
    good_countries = quality_df[
        (quality_df['missing_ratio'] <= quality_thresholds['missing_ratio']) &
        (quality_df['year_span'] >= quality_thresholds['year_span']) &
        (quality_df['data_points'] >= quality_thresholds['data_points']) &
        (quality_df['key_missing_ratio'] <= quality_thresholds['key_missing_ratio'])
    ].index
    
    # 输出数据质量分析结果
    print("\n数据质量分析结果:")
    print(f"总国家数: {len(df['country'].unique())}")
    print(f"满足质量要求的国家数: {len(good_countries)}")
    print("\n被排除的国家及其原因:")
    excluded_countries = set(df['country'].unique()) - set(good_countries)
    # for country in excluded_countries:
    #     print(f"\n{country}:")
    #     for metric, threshold in quality_thresholds.items():
    #         if quality_df.loc[country, metric] > threshold:
    #             print(f"- {metric}: {quality_df.loc[country, metric]:.2f} (阈值: {threshold})")
    
    # 筛选数据
    df_filtered = df[df['country'].isin(good_countries)].copy()
    
    # 输出筛选后的数据
    df_filtered.to_csv('2_data_after_quality_filter.csv', index=False)
    print("\n筛选后的数据已保存到 2_data_after_quality_filter.csv")
    
    return df_filtered

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
    
    # 输出原始数据
    df.to_csv('0_data_raw.csv', index=False)
    print("原始数据已保存到 0_data_raw.csv")
    
    # 第一步：过滤每个国家早期严重缺失数据的年份
    print("\n开始过滤早期严重缺失数据的年份...")
    df = filter_early_missing_data(df)
    df.to_csv('1_data_after_early_filter.csv', index=False)
    print("过滤早期缺失数据后的数据已保存到 1_data_after_early_filter.csv")
    
    # 第二步：分析数据质量并筛选出数据质量较好的国家
    print("\n开始分析数据质量并筛选国家...")
    df = analyze_data_quality(df)
    
    # 添加时序特征
    # 1. 滞后项特征
    for i in range(1, 6):
        df[f'renewable_share_lag_{i}'] = df.groupby('country')['renewables_share_energy'].shift(i)
    
    # 2. 移动平均特征
    df['renewable_ma_3'] = df.groupby('country')['renewables_share_energy'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['renewable_ma_5'] = df.groupby('country')['renewables_share_energy'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # 3. 趋势特征
    df['renewable_trend_3'] = df.groupby('country')['renewables_share_energy'].transform(
        lambda x: x.diff(3)
    )
    df['renewable_trend_5'] = df.groupby('country')['renewables_share_energy'].transform(
        lambda x: x.diff(5)
    )
    
    # 4. 计算化石能源消费变化率
    df['fossil_cons_change_pct'] = df.groupby('country')['fossil_share_energy'].transform(
        lambda x: x.pct_change(fill_method=None)
    )
    
    # 5. 添加政策相关特征
    df['paris_agreement'] = (df['year'] >= 2015).astype(int)
    df['years_since_paris'] = (df['year'] - 2015).clip(0)
    df['policy_strength'] = df['paris_agreement'] * df['years_since_paris']
    
    # 6. 添加技术发展特征
    df['renewable_tech_index'] = df['solar_share_elec'] + df['wind_share_elec']
    df['tech_growth_rate'] = df.groupby('country')['renewable_tech_index'].transform(
        lambda x: x.pct_change(fill_method=None)
    )
    
    # 7. 添加经济结构特征
    df['energy_intensity'] = df['energy_per_gdp'] * df['gdp']
    df['energy_per_capita'] = df['energy_per_gdp'] * df['gdp'] / df['population']
    
    # 8. 添加交互特征
    df['gdp_renewable_interaction'] = df['gdp'] * df['renewables_share_energy']
    df['population_renewable_interaction'] = df['population'] * df['renewables_share_energy']
    df['policy_economic_interaction'] = df['paris_agreement'] * df['gdp']
    
    # 9. 添加区域特征（使用ISO代码的前两位作为区域标识）
    df['region'] = df['iso_code'].str[:2]
    df['region_renewable_avg'] = df.groupby(['region', 'year'])['renewables_share_energy'].transform('mean')
    
    # 10. 添加可再生能源发展速度特征
    df['renewable_growth'] = df.groupby('country')['renewables_share_energy'].transform(
        lambda x: x.pct_change(fill_method=None)
    )
    
    return df

def prepare_features(df):
    """特征工程"""
    # 获取所有数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 按国家分别填充缺失值
    df = fill_missing_by_country(df, numeric_columns)
    
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
        max_depth=5,  # 降低树的深度，防止过拟合
        min_samples_split=10,  # 增加分裂所需的最小样本数
        min_samples_leaf=5,  # 增加叶节点所需的最小样本数
        max_features='sqrt',  # 使用sqrt(n_features)个特征进行分裂
        bootstrap=True,  # 使用bootstrap采样
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
        # 获取该国家在测试集中的数据
        country_mask_test = (df['country'] == country) & test_mask
        country_test_data = df[country_mask_test]
        
        # 确保该国家在测试集中有足够的数据点（至少3年）
        if len(country_test_data) >= 3:
            # 使用该国家在测试集中的数据进行预测
            X_test_country = X[country_mask_test]
            y_test_country = y[country_mask_test]
            
            # 计算该国家的R²分数
            score = model.score(X_test_country, y_test_country)
            country_scores[country] = score
    
    # 输出每个国家的预测性能
    print("\n各国预测性能 (R² 分数):")
    for country, score in sorted(country_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{country}: {score:.3f}")
    
    return model, X_train, X_test, y_train, y_test, country_scores, test_mask

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
        max_depth=5,  # 降低树的深度，防止过拟合
        min_samples_split=10,  # 增加分裂所需的最小样本数
        min_samples_leaf=5,  # 增加叶节点所需的最小样本数
        max_features='sqrt',  # 使用sqrt(n_features)个特征进行分裂
        bootstrap=True,  # 使用bootstrap采样
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
            # 确保训练集和测试集都有足够的样本
            if len(X_train) > 1 and len(X_test) > 1:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                cv_scores.append(score)
                print(f"Fold {fold} R² 分数: {score:.3f}")
            else:
                print(f"Fold {fold} 样本数量不足，跳过")
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
    # 获取该国家的最新数据
    country_data = df[df['country'] == country].copy()
    latest_data = country_data.iloc[-1:].copy()
    
    # 获取训练模型时使用的特征（数值类型特征）
    feature_columns = [col for col in df.columns 
                      if col not in ['country', 'year', 'renewables_share_energy', 'iso_code']]
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
    
    predictions = []
    current_data = latest_data.copy()
    
    # 计算历史趋势
    historical_trend = country_data['renewables_share_energy'].pct_change().mean()
    
    for i in range(years_to_predict):
        # 更新年份
        current_data['year'] = current_data['year'] + 1
        
        # 更新滞后特征
        for j in range(5, 0, -1):
            if j == 1:
                # 使用历史趋势来调整预测值
                if predictions:
                    current_data[f'renewable_share_lag_{j}'] = predictions[-1] * (1 + historical_trend)
                else:
                    current_data[f'renewable_share_lag_{j}'] = current_data['renewables_share_energy'].values[0]
            else:
                current_data[f'renewable_share_lag_{j}'] = current_data[f'renewable_share_lag_{j-1}']
        
        # 预测
        pred = model.predict(current_data[numeric_features])[0]
        
        # 确保预测值在合理范围内
        last_value = predictions[-1] if predictions else current_data['renewables_share_energy'].values[0]
        max_change = 0.1  # 最大允许变化率（10%）
        pred = np.clip(pred, last_value * (1 - max_change), last_value * (1 + max_change))
        
        predictions.append(pred)
    
    return predictions

def plot_predictions(country, historical_data, predictions, future_years):
    """绘制预测结果图表"""
    plt.figure(figsize=(12, 6))
    
    # 绘制历史数据
    plt.plot(historical_data['year'], historical_data['renewables_share_energy'], 
             label='Historical Data', marker='o')
    
    # 绘制预测数据
    future_years = range(historical_data['year'].max() + 1, 
                        historical_data['year'].max() + len(predictions) + 1)
    plt.plot(future_years, predictions, label='Prediction', marker='s', linestyle='--')
    
    plt.title(f'Renewable Energy Share Prediction - {country}')
    plt.xlabel('Year')
    plt.ylabel('Renewable Energy Share (%)')
    plt.legend()
    plt.grid(True)
    
    return plt

def plot_test_predictions(model, df, country, test_mask):
    """绘制测试集上的预测效果对比图"""
    # 获取该国家在测试集上的数据
    country_mask = (df['country'] == country) & test_mask
    country_test_data = df[country_mask]
    
    if len(country_test_data) < 3:
        return None
    
    # 准备特征
    feature_columns = [col for col in df.columns 
                      if col not in ['country', 'year', 'renewables_share_energy', 'iso_code']]
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
    
    # 获取测试集上的预测值
    X_test_country = df[country_mask][numeric_features]
    y_true = df[country_mask]['renewables_share_energy']
    y_pred = model.predict(X_test_country)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制实际值
    plt.plot(country_test_data['year'], y_true, 
             label='实际值', marker='o', color='blue')
    
    # 绘制预测值
    plt.plot(country_test_data['year'], y_pred, 
             label='预测值', marker='s', color='red', linestyle='--')
    
    # 计算R²分数
    r2_score = model.score(X_test_country, y_true)
    
    plt.title(f'{country} 测试集预测效果对比 (R² = {r2_score:.3f})')
    plt.xlabel('年份')
    plt.ylabel('可再生能源占比 (%)')
    plt.legend()
    plt.grid(True)
    
    return plt

def generate_model_report(model, df, X_train, X_test, y_train, y_test, country_scores, test_mask, cv_scores):
    """
    生成详细的模型评估报告
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import json
    
    report = {
        "基础模型信息": {
            "模型类型": "随机森林回归",
            "模型参数": model.get_params(),
            "特征数量": X_train.shape[1],
            "训练样本数": X_train.shape[0],
            "测试样本数": X_test.shape[0]
        },
        
        "时间范围信息": {
            "训练集": {
                "开始年份": int(df[df['year'] <= df['year'].max() - 5]['year'].min()),
                "结束年份": int(df[df['year'] <= df['year'].max() - 5]['year'].max())
            },
            "测试集": {
                "开始年份": int(df[df['year'] > df['year'].max() - 5]['year'].min()),
                "结束年份": int(df[df['year'] > df['year'].max() - 5]['year'].max())
            }
        },
        
        "整体模型性能": {
            "训练集": {
                "R2分数": float(model.score(X_train, y_train)),
                "均方误差(MSE)": float(mean_squared_error(y_train, model.predict(X_train))),
                "均方根误差(RMSE)": float(np.sqrt(mean_squared_error(y_train, model.predict(X_train)))),
                "平均绝对误差(MAE)": float(mean_absolute_error(y_train, model.predict(X_train)))
            },
            "测试集": {
                "R2分数": float(model.score(X_test, y_test)),
                "均方误差(MSE)": float(mean_squared_error(y_test, model.predict(X_test))),
                "均方根误差(RMSE)": float(np.sqrt(mean_squared_error(y_test, model.predict(X_test)))),
                "平均绝对误差(MAE)": float(mean_absolute_error(y_test, model.predict(X_test)))
            }
        },
        
        "交叉验证结果": {
            "折数": len(cv_scores),
            "平均R2分数": float(np.mean(cv_scores)),
            "R2分数标准差": float(np.std(cv_scores)),
            "各折分数": [float(score) for score in cv_scores]
        },
        
        "特征重要性": {
            feature: float(importance) 
            for feature, importance in zip(X_train.columns, model.feature_importances_)
        },
        
        "各国预测性能": {
            country: float(score) for country, score in country_scores.items()
        }
    }
    
    # 保存报告为JSON文件
    with open('model_evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    # 生成HTML报告
    html_report = f"""
    <html>
    <head>
        <title>可再生能源预测模型评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            .metric {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>可再生能源预测模型评估报告</h1>
        
        <h2>1. 基础模型信息</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>模型类型</td><td>随机森林回归</td></tr>
            <tr><td>特征数量</td><td>{X_train.shape[1]}</td></tr>
            <tr><td>训练样本数</td><td>{X_train.shape[0]}</td></tr>
            <tr><td>测试样本数</td><td>{X_test.shape[0]}</td></tr>
        </table>

        <h2>2. 时间范围信息</h2>
        <table>
            <tr><th>数据集</th><th>开始年份</th><th>结束年份</th></tr>
            <tr>
                <td>训练集</td>
                <td>{report['时间范围信息']['训练集']['开始年份']}</td>
                <td>{report['时间范围信息']['训练集']['结束年份']}</td>
            </tr>
            <tr>
                <td>测试集</td>
                <td>{report['时间范围信息']['测试集']['开始年份']}</td>
                <td>{report['时间范围信息']['测试集']['结束年份']}</td>
            </tr>
        </table>

        <h2>3. 模型性能指标</h2>
        <table>
            <tr><th>指标</th><th>训练集</th><th>测试集</th></tr>
            <tr>
                <td>R2分数</td>
                <td>{report['整体模型性能']['训练集']['R2分数']:.3f}</td>
                <td>{report['整体模型性能']['测试集']['R2分数']:.3f}</td>
            </tr>
            <tr>
                <td>均方误差(MSE)</td>
                <td>{report['整体模型性能']['训练集']['均方误差(MSE)']:.3f}</td>
                <td>{report['整体模型性能']['测试集']['均方误差(MSE)']:.3f}</td>
            </tr>
            <tr>
                <td>均方根误差(RMSE)</td>
                <td>{report['整体模型性能']['训练集']['均方根误差(RMSE)']:.3f}</td>
                <td>{report['整体模型性能']['测试集']['均方根误差(RMSE)']:.3f}</td>
            </tr>
            <tr>
                <td>平均绝对误差(MAE)</td>
                <td>{report['整体模型性能']['训练集']['平均绝对误差(MAE)']:.3f}</td>
                <td>{report['整体模型性能']['测试集']['平均绝对误差(MAE)']:.3f}</td>
            </tr>
        </table>

        <h2>4. 交叉验证结果</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>平均R2分数</td><td>{report['交叉验证结果']['平均R2分数']:.3f}</td></tr>
            <tr><td>R2分数标准差</td><td>{report['交叉验证结果']['R2分数标准差']:.3f}</td></tr>
        </table>

        <h2>5. 特征重要性（Top 10）</h2>
        <table>
            <tr><th>特征</th><th>重要性</th></tr>
            {
                ''.join([f"<tr><td>{feature}</td><td>{importance:.3f}</td></tr>"
                        for feature, importance in sorted(report['特征重要性'].items(),
                                                      key=lambda x: x[1], reverse=True)[:10]])
            }
        </table>

        <h2>6. 各国预测性能（Top 10）</h2>
        <table>
            <tr><th>国家</th><th>R2分数</th></tr>
            {
                ''.join([f"<tr><td>{country}</td><td>{score:.3f}</td></tr>"
                        for country, score in sorted(report['各国预测性能'].items(),
                                                 key=lambda x: x[1], reverse=True)[:10]])
            }
        </table>
    </body>
    </html>
    """
    
    # 保存HTML报告
    with open('model_evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    return report

def main():
    # 加载数据
    df = load_and_prepare_data('owid-energy-data.csv')
    
    # 分析数据质量并筛选国家
    print("\n开始数据质量分析...")
    df = analyze_data_quality(df)
    
    # 特征工程
    df = prepare_features(df)
    
    # 使用时间序列交叉验证评估模型
    print("\n执行时间序列交叉验证...")
    cv_scores = evaluate_with_time_series_cv(df)
    
    # 训练最终模型
    print("\n训练最终模型...")
    model, X_train, X_test, y_train, y_test, country_scores, test_mask = train_model(df, test_years=5)
    
    # 生成模型评估报告
    print("\n生成模型评估报告...")
    report = generate_model_report(model, df, X_train, X_test, y_train, y_test, country_scores, test_mask, cv_scores)
    print("模型评估报告已保存到 model_evaluation_report.json 和 model_evaluation_report.html")
    
    # 获取R²分数最高的五个国家
    top_countries = sorted(country_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # 为R²分数最高的五个国家绘制测试集预测效果对比图
    print("\n绘制测试集预测效果对比图...")
    for country, score in top_countries:
        plt = plot_test_predictions(model, df, country, test_mask)
        if plt is not None:
            plt.savefig(f'test_prediction_{country}.png')
            plt.close()
    
    # 为R²分数最高的五个国家进行未来预测
    for country, score in top_countries:
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