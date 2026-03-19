#!/usr/bin/env python3
"""
Step 2: CSV 推理结果特征提取
- 匹配100个已评分点位与436个CSV文件
- 提取关键统计特征（异常率、异常段数量、持续时间等）
- 全量436个点位特征提取
- 按传感器类型分类
"""
import pandas as pd
import numpy as np
import os
import json
import re
from collections import defaultdict

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(PROJ_ROOT, "data", "adtk_hbos_old")
CLEANED_DIR = os.path.join(PROJ_ROOT, "data", "cleaned")
FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

print("=" * 60)
print("Step 2: CSV 推理结果特征提取")
print("=" * 60)

# ============================================================
# 1. 建立文件名 → 点位名映射
# ============================================================
csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith('.csv')])
print(f"CSV 文件总数: {len(csv_files)}")

# 从文件名提取点位名
# 格式: global_adtk_hbos_m4_0.1_1200.0_{POINT_NAME}_20230718_trend_seasonal_resid.csv
def extract_point_name(filename):
    """从文件名提取点位名称"""
    # 删除前缀和后缀
    name = filename.replace("global_adtk_hbos_m4_0.1_1200.0_", "")
    name = name.replace("_20230718_trend_seasonal_resid.csv", "")
    return name

file_point_map = {}
for f in csv_files:
    point = extract_point_name(f)
    file_point_map[point] = f

# ============================================================
# 2. 匹配已评分点位
# ============================================================
evaluated_points = []
with open(os.path.join(CLEANED_DIR, "evaluated_points.txt"), "r") as fp:
    evaluated_points = [line.strip() for line in fp if line.strip()]

print(f"已评分点位数: {len(evaluated_points)}")

# 匹配
matched = []
unmatched = []
for ep in evaluated_points:
    if ep in file_point_map:
        matched.append((ep, file_point_map[ep]))
    else:
        # 尝试模糊匹配
        found = False
        for fp_name, fp_file in file_point_map.items():
            if ep.replace('.', '_') == fp_name.replace('.', '_') or ep in fp_name:
                matched.append((ep, fp_file))
                found = True
                break
        if not found:
            unmatched.append(ep)

print(f"匹配成功: {len(matched)}, 未匹配: {len(unmatched)}")
if unmatched:
    print(f"未匹配点位: {unmatched}")

# ============================================================
# 3. 特征提取函数
# ============================================================
def extract_features(filepath, point_name):
    """从单个 CSV 文件提取统计特征"""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  ⚠ 读取失败 {point_name}: {e}")
        return None
    
    # 获取传感器值列（第2列）
    sensor_col = [c for c in df.columns if c not in ['Time', 'global_mask', 'outlier_mask', 'local_mask', 'global_mask_cluster', 'local_mask_cluster']]
    sensor_col = sensor_col[0] if sensor_col else None
    
    features = {'point_name': point_name}
    
    # 基本统计
    features['total_rows'] = len(df)
    
    # 异常率统计
    for mask_col in ['global_mask', 'outlier_mask', 'local_mask']:
        if mask_col in df.columns:
            anomaly_count = int(df[mask_col].sum())
            features[f'{mask_col}_count'] = anomaly_count
            features[f'{mask_col}_ratio'] = round(anomaly_count / len(df), 6) if len(df) > 0 else 0
    
    # 异常段分析（基于 global_mask）
    if 'global_mask' in df.columns:
        mask = df['global_mask'].values
        # 计算连续异常段
        diff = np.diff(np.concatenate([[0], mask, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        num_clusters = len(starts)
        features['num_anomaly_clusters'] = num_clusters
        
        if num_clusters > 0:
            cluster_lengths = ends - starts
            features['avg_cluster_length'] = round(float(np.mean(cluster_lengths)), 2)
            features['max_cluster_length'] = int(np.max(cluster_lengths))
            features['min_cluster_length'] = int(np.min(cluster_lengths))
            features['median_cluster_length'] = round(float(np.median(cluster_lengths)), 2)
            features['std_cluster_length'] = round(float(np.std(cluster_lengths)), 2)
        else:
            features['avg_cluster_length'] = 0
            features['max_cluster_length'] = 0
            features['min_cluster_length'] = 0
            features['median_cluster_length'] = 0
            features['std_cluster_length'] = 0
    
    # 多标签一致性
    if all(c in df.columns for c in ['global_mask', 'outlier_mask', 'local_mask']):
        all_agree = ((df['global_mask'] == df['outlier_mask']) & (df['outlier_mask'] == df['local_mask'])).mean()
        features['label_agreement_rate'] = round(float(all_agree), 4)
        
        # 高置信异常率（三者都标记为异常）
        high_conf = ((df['global_mask'] == 1) & (df['outlier_mask'] == 1) & (df['local_mask'] == 1)).sum()
        features['high_confidence_anomaly_count'] = int(high_conf)
        features['high_confidence_anomaly_ratio'] = round(float(high_conf / len(df)), 6) if len(df) > 0 else 0
    
    # 传感器值统计
    if sensor_col:
        sensor_values = df[sensor_col].dropna()
        if len(sensor_values) > 0:
            features['sensor_mean'] = round(float(sensor_values.mean()), 4)
            features['sensor_std'] = round(float(sensor_values.std()), 4)
            features['sensor_min'] = round(float(sensor_values.min()), 4)
            features['sensor_max'] = round(float(sensor_values.max()), 4)
            features['sensor_range'] = round(float(sensor_values.max() - sensor_values.min()), 4)
            features['sensor_cv'] = round(float(sensor_values.std() / sensor_values.mean()), 4) if sensor_values.mean() != 0 else 0
    
    # 传感器类型分类
    features['sensor_type'] = classify_sensor(point_name)
    
    return features

def classify_sensor(point_name):
    """根据点位名称前缀分类传感器类型"""
    name = point_name.upper()
    if name.startswith(('TI_', 'TE_', 'TT', 'TV_', 'TY_', 'TLS_')):
        return 'Temperature'
    elif name.startswith(('PI_', 'PT_', 'PT2', 'PDI_', 'PDV_', 'PV_')):
        return 'Pressure'
    elif name.startswith(('FI_', 'FT_', 'FV_')):
        return 'Flow'
    elif name.startswith(('LI_', 'LV_', 'LDI_', 'LDV_')):
        return 'Level'
    elif name.startswith(('AI_', 'AT_', 'AT1')):
        return 'Analyzer'
    elif name.startswith('P6'):
        return 'Pump/Motor'
    elif name.startswith(('IP', 'IPC')):
        return 'Current/Power'
    elif name.startswith(('EI_', 'II_')):
        return 'Electrical'
    elif name.startswith(('HV_', 'HS', 'HSD_')):
        return 'Valve'
    elif name.startswith(('YH', 'YH0')):
        return 'SOV/Discrete'
    elif name.startswith(('MC_', 'MY_', 'MZ', 'MPC')):
        return 'Motor/Control'
    elif name.startswith(('COM', 'CS', 'C6', 'GDS', '6500')):
        return 'Communication/System'
    elif name.startswith('AC'):
        return 'AC_Controller'
    elif name.startswith('ST_'):
        return 'Speed'
    else:
        return 'Other'

# ============================================================
# 4. 提取已评分点位的特征
# ============================================================
print("\n--- 提取已评分点位特征 ---")
evaluated_features = []
for point_name, csv_file in matched:
    filepath = os.path.join(CSV_DIR, csv_file)
    feat = extract_features(filepath, point_name)
    if feat:
        evaluated_features.append(feat)
    if len(evaluated_features) % 20 == 0:
        print(f"  已处理 {len(evaluated_features)} / {len(matched)} ...")

df_eval_features = pd.DataFrame(evaluated_features)
eval_feat_path = os.path.join(FEATURES_DIR, "evaluated_points_features.csv")
df_eval_features.to_csv(eval_feat_path, index=False, encoding='utf-8-sig')
print(f"已评分点位特征已保存: {eval_feat_path} ({len(df_eval_features)} 行)")

# ============================================================
# 5. 提取全量点位特征
# ============================================================
print("\n--- 提取全量 436 个点位特征 ---")
all_features = []
for i, csv_file in enumerate(csv_files):
    point_name = extract_point_name(csv_file)
    filepath = os.path.join(CSV_DIR, csv_file)
    feat = extract_features(filepath, point_name)
    if feat:
        all_features.append(feat)
    if (i + 1) % 50 == 0:
        print(f"  已处理 {i+1} / {len(csv_files)} ...")

df_all_features = pd.DataFrame(all_features)
all_feat_path = os.path.join(FEATURES_DIR, "all_points_features.csv")
df_all_features.to_csv(all_feat_path, index=False, encoding='utf-8-sig')
print(f"全量点位特征已保存: {all_feat_path} ({len(df_all_features)} 行)")

# ============================================================
# 6. 传感器类型统计
# ============================================================
print("\n--- 传感器类型统计 ---")
type_dist = df_all_features['sensor_type'].value_counts()
for stype, count in type_dist.items():
    avg_anomaly = df_all_features[df_all_features['sensor_type'] == stype]['global_mask_ratio'].mean()
    print(f"  {stype:25s}: {count:3d} 个 | 平均异常率: {avg_anomaly*100:.2f}%")

# ============================================================
# 7. 保存统计摘要
# ============================================================
csv_summary = {
    "total_csv_files": len(csv_files),
    "matched_evaluated": len(matched),
    "unmatched_evaluated": len(unmatched),
    "unmatched_points": unmatched,
    "sensor_type_distribution": {k: int(v) for k, v in type_dist.items()},
    "global_stats": {
        "mean_anomaly_ratio": round(float(df_all_features['global_mask_ratio'].mean()), 4),
        "median_anomaly_ratio": round(float(df_all_features['global_mask_ratio'].median()), 4),
        "max_anomaly_ratio": round(float(df_all_features['global_mask_ratio'].max()), 4),
        "min_anomaly_ratio": round(float(df_all_features['global_mask_ratio'].min()), 4),
        "mean_num_clusters": round(float(df_all_features['num_anomaly_clusters'].mean()), 2),
    }
}

summary_path = os.path.join(FEATURES_DIR, "csv_features_summary.json")
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(csv_summary, f, ensure_ascii=False, indent=2)
print(f"\nCSV 特征摘要已保存: {summary_path}")
print("\n✅ CSV 推理结果特征提取完成！")
