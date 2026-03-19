#!/usr/bin/env python3
"""
Step 3: 数据关联分析与评审一致性计算
- 关联 Excel 评分与 CSV 推理特征
- 计算评审人一致性（Fleiss' Kappa, Krippendorff's Alpha）
- 按维度分层分析
- 输出综合评估报告
"""
import pandas as pd
import numpy as np
import os
import json
from itertools import combinations

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(PROJ_ROOT, "data", "cleaned")
FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Step 3: 数据关联分析与评审一致性分析")
print("=" * 60)

# ============================================================
# 1. 加载清洗后的数据
# ============================================================
df_scores = pd.read_csv(os.path.join(CLEANED_DIR, "scores_analysis.csv"))
df_eval_feat = pd.read_csv(os.path.join(FEATURES_DIR, "evaluated_points_features.csv"))
df_all_feat = pd.read_csv(os.path.join(FEATURES_DIR, "all_points_features.csv"))

print(f"评分数据: {df_scores.shape}")
print(f"已评分点位特征: {df_eval_feat.shape}")
print(f"全量点位特征: {df_all_feat.shape}")

# ============================================================
# 2. 关联评分与推理特征
# ============================================================
df_merged = pd.merge(df_scores, df_eval_feat, on='point_name', how='left')
merged_path = os.path.join(FEATURES_DIR, "merged_scores_features.csv")
df_merged.to_csv(merged_path, index=False, encoding='utf-8-sig')
print(f"\n关联后数据: {df_merged.shape}")
print(f"关联数据已保存: {merged_path}")

# ============================================================
# 3. 评审人一致性分析
# ============================================================
print("\n" + "=" * 60)
print("评审人一致性分析")
print("=" * 60)

# 评分结构
reviewers = ['dff', 'wyx', 'lzh', 'xpj', 'aym']
reviewer_names = {'dff': '窦丰丰', 'wyx': '王一雄', 'lzh': '梁泽华', 'xpj': '薛佩姣', 'aym': '安阳明'}
models = ['qwen', 'chatts', 'timer']
model_names = {'qwen': 'Qwen-VL-8B', 'chatts': 'ChatTS', 'timer': 'Timer'}

def compute_fleiss_kappa(ratings_matrix):
    """
    计算 Fleiss' Kappa
    ratings_matrix: (n_subjects, n_categories) - 每行是一个被评对象，每列是一个类别，值是该类别的评审人数
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # 假设每个对象的评审人数相同
    
    # 检查一致性
    if n_raters <= 1:
        return float('nan')
    
    # P_i: 每个对象上评审人之间的一致程度
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)
    
    # P_e: 期望一致程度
    p_j = np.sum(ratings_matrix, axis=0) / (n_subjects * n_raters)
    P_e = np.sum(p_j ** 2)
    
    if P_e == 1:
        return 1.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    return round(float(kappa), 4)

def compute_pairwise_agreement(scores_df, rater_cols):
    """计算配对一致率"""
    agreements = []
    for r1, r2 in combinations(rater_cols, 2):
        agree = (scores_df[r1] == scores_df[r2]).mean()
        agreements.append({
            'rater1': r1,
            'rater2': r2,
            'agreement': round(float(agree), 4)
        })
    return agreements

# 按模型计算一致性
consistency_results = {}
for model in models:
    model_cols = [f'{r}_{model}' for r in reviewers]
    model_scores = df_scores[model_cols].dropna()
    
    # 将评分映射到类别: 0 → 0, 0.5 → 1, 1 → 2
    categories = [0, 0.5, 1.0]
    n_categories = len(categories)
    n_subjects = len(model_scores)
    
    # 构建评分矩阵
    ratings_matrix = np.zeros((n_subjects, n_categories))
    for i in range(n_subjects):
        for col in model_cols:
            val = model_scores.iloc[i][col]
            if val in categories:
                cat_idx = categories.index(val)
                ratings_matrix[i, cat_idx] += 1
    
    kappa = compute_fleiss_kappa(ratings_matrix)
    
    # 配对一致率
    pairwise = compute_pairwise_agreement(model_scores, model_cols)
    avg_agreement = np.mean([p['agreement'] for p in pairwise])
    
    consistency_results[model] = {
        'fleiss_kappa': kappa,
        'avg_pairwise_agreement': round(float(avg_agreement), 4),
        'pairwise_details': pairwise,
    }
    
    print(f"\n{model_names[model]}:")
    print(f"  Fleiss' Kappa = {kappa}")
    print(f"  平均配对一致率 = {avg_agreement:.4f}")

# ============================================================
# 4. 模型综合评估报告
# ============================================================
print("\n" + "=" * 60)
print("模型综合评估报告")
print("=" * 60)

model_report = {}
for model in models:
    avg_col = f'avg_{model}'
    model_cols = [f'{r}_{model}' for r in reviewers]
    
    all_scores = df_scores[model_cols].values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)]
    
    report = {
        'mean_score': round(float(np.mean(all_scores)), 4),
        'median_score': round(float(np.median(all_scores)), 4),
        'std_score': round(float(np.std(all_scores)), 4),
        'score_distribution': {
            'score_0': int(np.sum(all_scores == 0)),
            'score_0.5': int(np.sum(all_scores == 0.5)),
            'score_1': int(np.sum(all_scores == 1.0)),
        },
        'score_0_rate': round(float(np.mean(all_scores == 0)), 4),
        'score_0.5_rate': round(float(np.mean(all_scores == 0.5)), 4),
        'score_1_rate': round(float(np.mean(all_scores == 1.0)), 4),
        'pass_rate': round(float(np.mean(all_scores >= 0.5)), 4),
    }
    model_report[model_names[model]] = report
    
    print(f"\n{model_names[model]}:")
    print(f"  总评分数: {len(all_scores)}")
    print(f"  均分: {report['mean_score']}")
    print(f"  评分分布: 0分={report['score_distribution']['score_0']}, 0.5分={report['score_distribution']['score_0.5']}, 1分={report['score_distribution']['score_1']}")
    print(f"  通过率(≥0.5): {report['pass_rate']*100:.1f}%")

# ============================================================
# 5. 按评价类别分层分析
# ============================================================
print("\n" + "=" * 60)
print("按评价类别分层分析")
print("=" * 60)

category_analysis = {}
for cat in df_scores['eval_category'].unique():
    mask = df_scores['eval_category'] == cat
    cat_data = df_scores[mask]
    
    cat_result = {'count': int(len(cat_data))}
    for model in models:
        avg_col = f'avg_{model}'
        if avg_col in cat_data.columns:
            cat_result[model_names[model]] = round(float(cat_data[avg_col].mean()), 4)
    
    category_analysis[cat] = cat_result
    
    scores_str = ", ".join([f"{model_names[m]}={cat_result.get(model_names[m], 'N/A')}" for m in models])
    print(f"  {cat:25s} (n={len(cat_data):2d}): {scores_str}")

# ============================================================
# 6. 传感器类型 × 模型评分分析
# ============================================================
print("\n" + "=" * 60)
print("传感器类型 × 模型评分（已评估点位）")
print("=" * 60)

sensor_model_analysis = {}
if 'sensor_type' in df_merged.columns:
    for stype in sorted(df_merged['sensor_type'].dropna().unique()):
        mask = df_merged['sensor_type'] == stype
        sdata = df_merged[mask]
        
        result = {'count': int(len(sdata))}
        for model in models:
            avg_col = f'avg_{model}'
            if avg_col in sdata.columns:
                result[model_names[model]] = round(float(sdata[avg_col].mean()), 4)
        
        sensor_model_analysis[stype] = result
        scores_str = ", ".join([f"{model_names[m]}={result.get(model_names[m], 'N/A')}" for m in models])
        print(f"  {stype:25s} (n={len(sdata):2d}): {scores_str}")

# ============================================================
# 7. 评分与异常率相关性分析
# ============================================================
print("\n" + "=" * 60)
print("评分 vs 异常率相关性")
print("=" * 60)

if 'global_mask_ratio' in df_merged.columns:
    for model in models:
        avg_col = f'avg_{model}'
        valid_mask = df_merged[avg_col].notna() & df_merged['global_mask_ratio'].notna()
        if valid_mask.sum() > 2:
            corr = df_merged.loc[valid_mask, avg_col].corr(df_merged.loc[valid_mask, 'global_mask_ratio'])
            print(f"  {model_names[model]} vs 异常率: r = {corr:.4f}")

# ============================================================
# 8. 保存综合报告
# ============================================================
comprehensive_report = {
    "data_overview": {
        "total_evaluated_points": int(len(df_scores)),
        "total_csv_points": int(len(df_all_feat)),
        "matched_points": int(len(df_merged.dropna(subset=['total_rows']))),
        "reviewers": 5,
        "models": 3,
    },
    "model_report": model_report,
    "reviewer_consistency": {
        model_names[m]: {
            "fleiss_kappa": consistency_results[m]['fleiss_kappa'],
            "avg_pairwise_agreement": consistency_results[m]['avg_pairwise_agreement'],
        }
        for m in models
    },
    "category_analysis": category_analysis,
    "sensor_type_analysis": sensor_model_analysis,
}

report_path = os.path.join(RESULTS_DIR, "comprehensive_evaluation_report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
print(f"\n综合评估报告已保存: {report_path}")

# ============================================================
# 9. 生成模型排名表（Markdown 格式）
# ============================================================
ranking_md = os.path.join(RESULTS_DIR, "model_ranking.md")
with open(ranking_md, 'w', encoding='utf-8') as f:
    f.write("# 异常检测模型评估排名\n\n")
    f.write(f"*数据基于 {len(df_scores)} 个点位 × 5 位评审人的人工评分*\n\n")
    
    f.write("## 模型综合排名\n\n")
    f.write("| 排名 | 模型 | 均分 | 中位分 | 通过率(≥0.5) | 良好率(=1) | 差评率(=0) |\n")
    f.write("|------|------|------|--------|-------------|-----------|----------|\n")
    
    sorted_models = sorted(model_report.items(), key=lambda x: x[1]['mean_score'], reverse=True)
    for i, (mname, mdata) in enumerate(sorted_models, 1):
        f.write(f"| {i} | {mname} | {mdata['mean_score']:.3f} | {mdata['median_score']:.3f} | {mdata['pass_rate']*100:.1f}% | {mdata['score_1_rate']*100:.1f}% | {mdata['score_0_rate']*100:.1f}% |\n")
    
    f.write("\n## 评审人一致性\n\n")
    f.write("| 模型 | Fleiss' Kappa | 平均配对一致率 |\n")
    f.write("|------|-------------|---------------|\n")
    for m in models:
        r = consistency_results[m]
        f.write(f"| {model_names[m]} | {r['fleiss_kappa']:.4f} | {r['avg_pairwise_agreement']*100:.1f}% |\n")
    
    f.write("\n## 各评审人评分统计\n\n")
    f.write("| 评审人 | Qwen-VL-8B | ChatTS | Timer | 总平均分 |\n")
    f.write("|--------|-----------|--------|-------|----------|\n")
    for r in reviewers:
        q_mean = df_scores[f'{r}_qwen'].mean()
        c_mean = df_scores[f'{r}_chatts'].mean()
        t_mean = df_scores[f'{r}_timer'].mean()
        avg = np.mean([q_mean, c_mean, t_mean])
        f.write(f"| {reviewer_names[r]} | {q_mean:.3f} | {c_mean:.3f} | {t_mean:.3f} | {avg:.3f} |\n")
    
    f.write("\n## 按评价类别分层\n\n")
    f.write("| 评价类别 | 数量 | Qwen-VL-8B | ChatTS | Timer |\n")
    f.write("|---------|------|-----------|--------|-------|\n")
    for cat, cdata in sorted(category_analysis.items(), key=lambda x: x[1]['count'], reverse=True):
        q = cdata.get('Qwen-VL-8B', '-')
        c = cdata.get('ChatTS', '-')
        t = cdata.get('Timer', '-')
        f.write(f"| {cat} | {cdata['count']} | {q} | {c} | {t} |\n")
    
    if sensor_model_analysis:
        f.write("\n## 按传感器类型分层\n\n")
        f.write("| 传感器类型 | 数量 | Qwen-VL-8B | ChatTS | Timer |\n")
        f.write("|-----------|------|-----------|--------|-------|\n")
        for stype, sdata in sorted(sensor_model_analysis.items(), key=lambda x: x[1]['count'], reverse=True):
            q = sdata.get('Qwen-VL-8B', '-')
            c = sdata.get('ChatTS', '-')
            t = sdata.get('Timer', '-')
            f.write(f"| {stype} | {sdata['count']} | {q} | {c} | {t} |\n")

print(f"模型排名报告已保存: {ranking_md}")
print("\n✅ 数据关联分析与评审一致性分析完成！")
