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
    
    # 11. 添加分组特征
    # 计算每个国家的平均GDP和人均GDP
    country_stats = df.groupby('country').agg({
        'gdp': 'mean',
        'population': 'mean'
    }).reset_index()
    
    # 计算人均GDP
    country_stats['gdp_per_capita'] = country_stats['gdp'] / country_stats['population']
    
    # 计算GDP和人均GDP的分位数
    gdp_median = country_stats['gdp'].median()
    gdp_per_capita_median = country_stats['gdp_per_capita'].median()
    
    # 计算每个国家的平均能源结构
    energy_stats = df.groupby('country').agg({
        'fossil_share_energy': 'mean',
        'renewables_share_energy': 'mean'
    }).reset_index()
    
    # 计算化石能源和可再生能源占比的分位数
    fossil_median = energy_stats['fossil_share_energy'].median()
    renewable_median = energy_stats['renewables_share_energy'].median()
    
    # 创建分组
    country_stats['economic_group'] = country_stats.apply(
        lambda x: 'high_income' if x['gdp_per_capita'] > gdp_per_capita_median else 'low_income',
        axis=1
    )
    
    energy_stats['energy_group'] = energy_stats.apply(
        lambda x: 'renewable_leading' if x['renewables_share_energy'] > renewable_median else 'fossil_dependent',
        axis=1
    )
    
    # 将分组信息合并回主数据框
    df = df.merge(country_stats[['country', 'economic_group']], on='country', how='left')
    df = df.merge(energy_stats[['country', 'energy_group']], on='country', how='left')
    
    # 创建组合分组
    df['combined_group'] = df['economic_group'] + '_' + df['energy_group']
    
    # 输出分组统计信息
    print("\n分组统计信息:")
    print("\n经济发展水平分组:")
    print(df['economic_group'].value_counts())
    print("\n能源结构分组:")
    print(df['energy_group'].value_counts())
    print("\n组合分组:")
    print(df['combined_group'].value_counts())
    
    return df

class FeatureProcessor:
    """特征处理器类，用于管理特征工程过程"""
    def __init__(self):
        self.scalers = {}  # 存储每个特征的标准化器
        self.feature_stats = {}  # 存储特征统计信息
        self.feature_bounds = {}  # 存储特征边界值
    
    def fit(self, df, numeric_columns):
        """
        拟合数据，计算必要的统计信息
        
        Args:
            df: 数据框
            numeric_columns: 数值类型列名列表
        """
        self.numeric_columns = numeric_columns
        
        # 计算并存储每个特征的统计信息
        for col in numeric_columns:
            # 计算基本统计量
            stats = df[col].describe()
            self.feature_stats[col] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'median': df[col].median()
            }
            
            # 根据特征分布确定异常值边界
            # 使用Tukey方法计算边界
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 存储边界值
            self.feature_bounds[col] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
            # 创建并拟合标准化器
            scaler = StandardScaler()
            # 使用非异常值来拟合标准化器
            valid_data = df[col][(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            if len(valid_data) > 0:  # 确保有足够的有效数据
                scaler.fit(valid_data.values.reshape(-1, 1))
                self.scalers[col] = scaler
    
    def transform(self, df):
        """
        转换数据
        
        Args:
            df: 数据框
        
        Returns:
            转换后的数据框
        """
        df_transformed = df.copy()
        
        # 1. 处理缺失值
        df_transformed = self._handle_missing_values(df_transformed)
        
        # 2. 处理异常值
        df_transformed = self._handle_outliers(df_transformed)
        
        # 3. 标准化
        df_transformed = self._standardize(df_transformed)
        
        return df_transformed
    
    def _handle_missing_values(self, df):
        """处理缺失值"""
        df_filled = df.copy()
        
        for col in self.numeric_columns:
            # 对每个国家分别处理
            for country in df['country'].unique():
                country_mask = df['country'] == country
                country_data = df.loc[country_mask, col]
                
                if not country_data.isna().all():
                    # 使用该国家的中位数填充
                    median_value = country_data.median()
                    df_filled.loc[country_mask, col] = country_data.fillna(median_value)
                else:
                    # 使用全局中位数填充
                    df_filled.loc[country_mask, col] = df_filled[col].fillna(self.feature_stats[col]['median'])
        
        return df_filled
    
    def _handle_outliers(self, df):
        """处理异常值"""
        df_cleaned = df.copy()
        
        for col in self.numeric_columns:
            bounds = self.feature_bounds[col]
            # 使用边界值进行截断
            df_cleaned[col] = df_cleaned[col].clip(bounds['lower'], bounds['upper'])
        
        return df_cleaned
    
    def _standardize(self, df):
        """标准化数据"""
        df_scaled = df.copy()
        
        for col in self.numeric_columns:
            if col in self.scalers:
                # 使用已经拟合的标准化器进行转换
                df_scaled[col] = self.scalers[col].transform(df_scaled[col].values.reshape(-1, 1)).ravel()
        
        return df_scaled
    
    def save(self, filepath):
        """保存特征处理器的状态"""
        import joblib
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """加载特征处理器的状态"""
        import joblib
        return joblib.load(filepath)

def prepare_features(df):
    """
    特征工程
    
    Args:
        df: 数据框
    
    Returns:
        转换后的数据框和特征处理器
    """
    # 获取所有数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 创建并拟合特征处理器
    processor = FeatureProcessor()
    processor.fit(df, numeric_columns)
    
    # 转换数据
    df_transformed = processor.transform(df)
    
    # 保存特征处理器
    processor.save('feature_processor.joblib')
    
    # 保存处理后的数据
    df_transformed.to_csv('data_processed.csv', index=False)
    
    # 输出数据处理报告
    print("\n数据处理报告:")
    for col in numeric_columns:
        print(f"\n特征 {col}:")
        print(f"原始数据范围: [{df[col].min():.2f}, {df[col].max():.2f}]")
        print(f"处理后数据范围: [{df_transformed[col].min():.2f}, {df_transformed[col].max():.2f}]")
        print(f"缺失值比例: {df[col].isna().mean():.2%}")
        print(f"异常值边界: [{processor.feature_bounds[col]['lower']:.2f}, {processor.feature_bounds[col]['upper']:.2f}]")
    
    return df_transformed, processor

def train_model(df, test_years=5):
    """
    训练随机森林模型，使用时间序列划分方法，支持分组训练
    
    Args:
        df: 数据框
        test_years: 用作测试的年份数量
    """
    # 准备特征，排除非数值列
    feature_columns = [col for col in df.columns 
                      if col not in ['country', 'year', 'renewables_share_energy', 'iso_code',
                                   'economic_group', 'energy_group', 'combined_group']]
    
    # 确保所有特征都是数值类型
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
    
    # 为每个分组训练单独的模型
    models = {}
    group_scores = {}
    
    # 创建全局的测试集掩码
    cutoff_year = df['year'].max() - test_years
    test_mask = df['year'] > cutoff_year
    
    for group in df['combined_group'].unique():
        print(f"\n训练 {group} 组的模型...")
        
        # 获取该组的所有国家
        group_countries = df[df['combined_group'] == group]['country'].unique()
        
        # 存储每个国家的训练和测试数据
        X_train_group = []
        X_test_group = []
        y_train_group = []
        y_test_group = []
        
        # 对每个国家分别处理
        for country in group_countries:
            country_data = df[df['country'] == country].copy()
            
            # 按时间排序
            country_data = country_data.sort_values('year')
            
            # 划分训练集和测试集
            cutoff_year = country_data['year'].max() - test_years
            train_data = country_data[country_data['year'] <= cutoff_year]
            test_data = country_data[country_data['year'] > cutoff_year]
            
            if len(train_data) > 0 and len(test_data) > 0:
                # 准备特征矩阵
                X_train_country = train_data[numeric_features]
                X_test_country = test_data[numeric_features]
                y_train_country = train_data['renewables_share_energy']
                y_test_country = test_data['renewables_share_energy']
                
                # 确保没有无穷大值
                X_train_country = X_train_country.replace([np.inf, -np.inf], np.nan)
                X_test_country = X_test_country.replace([np.inf, -np.inf], np.nan)
                
                # 填充缺失值
                X_train_country = fill_missing_by_country(pd.concat([train_data[['country']], X_train_country], axis=1), numeric_features)[numeric_features]
                X_test_country = fill_missing_by_country(pd.concat([test_data[['country']], X_test_country], axis=1), numeric_features)[numeric_features]
                
                # 添加到组数据中
                X_train_group.append(X_train_country)
                X_test_group.append(X_test_country)
                y_train_group.append(y_train_country)
                y_test_group.append(y_test_country)
        
        if len(X_train_group) > 0 and len(X_test_group) > 0:
            # 合并所有国家的数据
            X_train_group = pd.concat(X_train_group)
            X_test_group = pd.concat(X_test_group)
            y_train_group = pd.concat(y_train_group)
            y_test_group = pd.concat(y_test_group)
            
            # 训练模型
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            
            try:
                model.fit(X_train_group, y_train_group)
                models[group] = model
                
                # 计算该组的性能
                train_score = model.score(X_train_group, y_train_group)
                test_score = model.score(X_test_group, y_test_group)
                
                group_scores[group] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'n_countries': len(group_countries),
                    'n_samples_train': len(X_train_group),
                    'n_samples_test': len(X_test_group)
                }
                
                print(f"{group} 组性能:")
                print(f"国家数量: {len(group_countries)}")
                print(f"训练集 R²: {train_score:.3f}")
                print(f"测试集 R²: {test_score:.3f}")
                print(f"训练样本数: {len(X_train_group)}")
                print(f"测试样本数: {len(X_test_group)}")
                
            except Exception as e:
                print(f"{group} 组训练失败: {str(e)}")
                continue
    
    # 计算每个国家的预测性能
    countries = df['country'].unique()
    country_scores = {}
    
    for country in countries:
        # 获取该国家在测试集中的数据
        country_data = df[df['country'] == country].copy()
        country_data = country_data.sort_values('year')
        
        # 划分训练集和测试集
        cutoff_year = country_data['year'].max() - test_years
        test_data = country_data[country_data['year'] > cutoff_year]
        
        # 确保该国家在测试集中有足够的数据点（至少3年）
        if len(test_data) >= 3:
            # 获取该国家所属的组
            country_group = test_data['combined_group'].iloc[0]
            
            # 如果该组有对应的模型
            if country_group in models:
                # 准备测试数据
                X_test_country = test_data[numeric_features]
                X_test_country = X_test_country.replace([np.inf, -np.inf], np.nan)
                X_test_country = fill_missing_by_country(pd.concat([test_data[['country']], X_test_country], axis=1), numeric_features)[numeric_features]
                y_test_country = test_data['renewables_share_energy']
                
                # 使用对应组的模型进行预测
                model = models[country_group]
                
                # 计算该国家的R²分数
                score = model.score(X_test_country, y_test_country)
                country_scores[country] = score
    
    # 输出每个国家的预测性能
    print("\n各国预测性能 (R² 分数):")
    for country, score in sorted(country_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{country}: {score:.3f}")
    
    return models, X_train_group, X_test_group, y_train_group, y_test_group, country_scores, test_mask, group_scores

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

def main():
    # 加载数据
    df = load_and_prepare_data('owid-energy-data.csv')
    
    # 分析数据质量并筛选国家
    print("\n开始数据质量分析...")
    df = analyze_data_quality(df)
    
    # 特征工程
    df, processor = prepare_features(df)
    
    # 使用时间序列交叉验证评估模型
    print("\n执行时间序列交叉验证...")
    cv_scores = evaluate_with_time_series_cv(df)
    
    # 训练最终模型
    print("\n训练最终模型...")
    models, X_train_group, X_test_group, y_train_group, y_test_group, country_scores, test_mask, group_scores = train_model(df, test_years=5)
    
    # 输出整体模型性能
    print("\n各分组模型性能:")
    for group, scores in group_scores.items():
        print(f"\n{group}组:")
        print(f"国家数量: {scores['n_countries']}")
        print(f"训练集 R²: {scores['train_score']:.3f}")
        print(f"测试集 R²: {scores['test_score']:.3f}")
        print(f"训练样本数: {scores['n_samples_train']}")
        print(f"测试样本数: {scores['n_samples_test']}")
    
    # 获取R²分数最高的五个国家
    top_countries = sorted(country_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nR²分数最高的五个国家:")
    for country, score in top_countries:
        print(f"{country}: {score:.3f}")
    
    # 为R²分数最高的五个国家绘制测试集预测效果对比图
    print("\n绘制R²分数最高的五个国家的测试集预测效果对比图...")
    for country, score in top_countries:
        plt = plot_test_predictions(models[df[df['country'] == country]['combined_group'].iloc[0]], df, country, test_mask)
        if plt is not None:
            plt.savefig(f'test_prediction_top_{country}.png')
            plt.close()
    
    # 为R²分数最高的五个国家进行未来预测
    for country, score in top_countries:
        country_group = df[df['country'] == country]['combined_group'].iloc[0]
        predictions = predict_future(models[country_group], df, country, years_to_predict=10)
        historical_data = df[df['country'] == country]
        
        print(f"\n{country}未来10年可再生能源占比预测:")
        for year, pred in enumerate(predictions, start=historical_data['year'].max() + 1):
            print(f"{year}年: {pred:.2f}%")
        
        # 绘制预测图表
        plt = plot_predictions(country, historical_data, predictions, range(len(predictions)))
        plt.savefig(f'prediction_top_{country}.png')
        plt.close()
    
    # 获取R²分数最差的五个国家
    bottom_countries = sorted(country_scores.items(), key=lambda x: x[1])[:5]
    print("\nR²分数最差的五个国家:")
    for country, score in bottom_countries:
        print(f"{country}: {score:.3f}")
    
    # 为R²分数最差的五个国家绘制测试集预测效果对比图
    print("\n绘制R²分数最差的五个国家的测试集预测效果对比图...")
    for country, score in bottom_countries:
        plt = plot_test_predictions(models[df[df['country'] == country]['combined_group'].iloc[0]], df, country, test_mask)
        if plt is not None:
            plt.savefig(f'test_prediction_bottom_{country}.png')
            plt.close()
    
    # 为R²分数最差的五个国家进行未来预测
    for country, score in bottom_countries:
        country_group = df[df['country'] == country]['combined_group'].iloc[0]
        predictions = predict_future(models[country_group], df, country, years_to_predict=10)
        historical_data = df[df['country'] == country]
        
        print(f"\n{country}未来10年可再生能源占比预测:")
        for year, pred in enumerate(predictions, start=historical_data['year'].max() + 1):
            print(f"{year}年: {pred:.2f}%")
        
        # 绘制预测图表
        plt = plot_predictions(country, historical_data, predictions, range(len(predictions)))
        plt.savefig(f'prediction_bottom_{country}.png')
        plt.close()

if __name__ == "__main__":
    main()