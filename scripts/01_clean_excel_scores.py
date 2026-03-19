#!/usr/bin/env python3
"""
Step 1: 清洗 Excel 评分文件
- 去除标题行和汇总行
- 重命名列为语义化名称
- 提取人工评价分类标签
- 输出结构化 CSV
"""
import pandas as pd
import numpy as np
import os
import json

# ============================================================
# 1. 读取原始 Excel
# ============================================================
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(PROJ_ROOT, "docs", "异常检测算法评分.xlsx")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "data", "cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: 清洗 Excel 评分文件")
print("=" * 60)

df_raw = pd.read_excel(EXCEL_PATH)
print(f"原始文件形状: {df_raw.shape}")

# ============================================================
# 2. 去除第0行(子标题行)和最后1行(汇总行)
# ============================================================
# 第0行是子标题（数据集, 测试集, Qwen-VL-8B, ...）
# 最后一行是分数统计汇总
df = df_raw.iloc[1:-1].reset_index(drop=True)
print(f"去除标题/汇总行后: {df.shape}")

# ============================================================
# 3. 重命名列
# ============================================================
column_mapping = {
    'Unnamed: 0': 'point_name',
    'Unnamed: 1': 'is_test',
    'Unnamed: 2': 'img_qwen_vl',
    'Unnamed: 3': 'img_chatts',
    'Unnamed: 4': 'img_timer',
    'Unnamed: 5': 'img_adtk_hbos',
    'Unnamed: 6': 'human_eval_1',
    'Unnamed: 7': 'human_eval_2',
    'Unnamed: 8': 'has_disagreement',
    '窦丰丰评分': 'dff_qwen',
    'Unnamed: 10': 'dff_chatts',
    'Unnamed: 11': 'dff_timer',
    '王一雄评分': 'wyx_qwen',
    'Unnamed: 13': 'wyx_chatts',
    'Unnamed: 14': 'wyx_timer',
    '梁泽华评分': 'lzh_qwen',
    'Unnamed: 16': 'lzh_chatts',
    'Unnamed: 17': 'lzh_timer',
    '薛佩姣评分': 'xpj_qwen',
    'Unnamed: 19': 'xpj_chatts',
    'Unnamed: 20': 'xpj_timer',
    '安阳明评分': 'aym_qwen',
    'Unnamed: 22': 'aym_chatts',
    'Unnamed: 23': 'aym_timer',
}
df = df.rename(columns=column_mapping)

# ============================================================
# 4. 类型转换
# ============================================================
score_cols = [
    'dff_qwen', 'dff_chatts', 'dff_timer',
    'wyx_qwen', 'wyx_chatts', 'wyx_timer',
    'lzh_qwen', 'lzh_chatts', 'lzh_timer',
    'xpj_qwen', 'xpj_chatts', 'xpj_timer',
    'aym_qwen', 'aym_chatts', 'aym_timer',
]

for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================
# 5. 计算模型平均分（每个点位，先按评审人平均）
# ============================================================
qwen_cols = ['dff_qwen', 'wyx_qwen', 'lzh_qwen', 'xpj_qwen', 'aym_qwen']
chatts_cols = ['dff_chatts', 'wyx_chatts', 'lzh_chatts', 'xpj_chatts', 'aym_chatts']
timer_cols = ['dff_timer', 'wyx_timer', 'lzh_timer', 'xpj_timer', 'aym_timer']

df['avg_qwen'] = df[qwen_cols].mean(axis=1)
df['avg_chatts'] = df[chatts_cols].mean(axis=1)
df['avg_timer'] = df[timer_cols].mean(axis=1)

# ============================================================
# 6. 提取人工评价分类标签
# ============================================================
def classify_eval(text):
    """将人工评价文本归类为标签"""
    if pd.isna(text):
        return 'unknown'
    text = str(text).strip()
    if '效果好' in text:
        return 'good'
    if '漏标' in text or '漏检' in text:
        return 'under_detection'
    if '多标' in text or '过度检测' in text or '过度标注' in text:
        return 'over_detection'
    if '正常检测' in text or '正常数据' in text:
        return 'normal'
    if '部分未识别' in text or '只识别出部分' in text or '部分尖峰' in text:
        return 'partial_detection'
    if '同类数据很多' in text or '只标记了部分' in text:
        return 'inconsistent_labeling'
    if '未识别' in text or '未检测' in text or '没有检测' in text or '没有识别' in text:
        return 'missed_detection'
    if '正常工况' in text:
        return 'false_positive_concern'
    if '数据问题' in text:
        return 'data_issue'
    if '正常的数据波动' in text or '似乎没有工况变化' in text:
        return 'normal'
    return 'other'

df['eval_category'] = df['human_eval_1'].apply(classify_eval)

# ============================================================
# 7. 提取图片 ID（用于后续关联）
# ============================================================
def extract_img_id(formula):
    """从 DISPIMG 公式中提取图片 ID"""
    if pd.isna(formula):
        return None
    import re
    match = re.search(r'ID_([A-F0-9]+)', str(formula))
    return match.group(1) if match else None

for img_col in ['img_qwen_vl', 'img_chatts', 'img_timer', 'img_adtk_hbos']:
    df[f'{img_col}_id'] = df[img_col].apply(extract_img_id)

# ============================================================
# 8. 输出结构化数据
# ============================================================
# 8a. 完整清洗后数据
output_full = os.path.join(OUTPUT_DIR, "scores_cleaned_full.csv")
df.to_csv(output_full, index=False, encoding='utf-8-sig')
print(f"完整清洗数据已保存: {output_full}")

# 8b. 仅评分数据（便于分析）
score_analysis_cols = ['point_name', 'eval_category', 'human_eval_1'] + score_cols + ['avg_qwen', 'avg_chatts', 'avg_timer']
df_scores = df[score_analysis_cols]
output_scores = os.path.join(OUTPUT_DIR, "scores_analysis.csv")
df_scores.to_csv(output_scores, index=False, encoding='utf-8-sig')
print(f"评分分析数据已保存: {output_scores}")

# 8c. 点位列表（用于和 CSV 关联）
output_points = os.path.join(OUTPUT_DIR, "evaluated_points.txt")
with open(output_points, 'w') as f:
    for p in df['point_name'].values:
        f.write(f"{p}\n")
print(f"已评估点位列表已保存: {output_points}")

# ============================================================
# 9. 打印统计摘要
# ============================================================
print("\n" + "=" * 60)
print("数据摘要")
print("=" * 60)
print(f"有效点位数: {len(df)}")
print(f"评审人数: 5 (窦丰丰, 王一雄, 梁泽华, 薛佩姣, 安阳明)")
print(f"被评模型: 3 (Qwen-VL-8B, ChatTS, Timer)")

print("\n--- 模型综合评分 ---")
for model, avg_col in [('Qwen-VL-8B', 'avg_qwen'), ('ChatTS', 'avg_chatts'), ('Timer', 'avg_timer')]:
    mean_score = df[avg_col].mean()
    median_score = df[avg_col].median()
    good_rate = (df[avg_col] >= 0.7).mean() * 100
    bad_rate = (df[avg_col] <= 0.2).mean() * 100
    print(f"  {model:12s}: 均分={mean_score:.3f}, 中位数={median_score:.3f}, 良好率={good_rate:.1f}%, 差评率={bad_rate:.1f}%")

print("\n--- 人工评价分类分布 ---")
eval_dist = df['eval_category'].value_counts()
for cat, count in eval_dist.items():
    print(f"  {cat:25s}: {count} ({count/len(df)*100:.1f}%)")

print("\n--- 各评审人评分统计 ---")
reviewers = {
    '窦丰丰': ('dff_qwen', 'dff_chatts', 'dff_timer'),
    '王一雄': ('wyx_qwen', 'wyx_chatts', 'wyx_timer'),
    '梁泽华': ('lzh_qwen', 'lzh_chatts', 'lzh_timer'),
    '薛佩姣': ('xpj_qwen', 'xpj_chatts', 'xpj_timer'),
    '安阳明': ('aym_qwen', 'aym_chatts', 'aym_timer'),
}
for name, (q, c, t) in reviewers.items():
    avg = df[[q, c, t]].mean().mean()
    print(f"  {name}: 总体平均分={avg:.3f} | Qwen={df[q].mean():.3f}, ChatTS={df[c].mean():.3f}, Timer={df[t].mean():.3f}")

# ============================================================
# 10. 保存统计摘要 JSON
# ============================================================
summary = {
    "total_points": int(len(df)),
    "total_reviewers": 5,
    "total_models": 3,
    "models": {
        "Qwen-VL-8B": {
            "mean_score": round(float(df['avg_qwen'].mean()), 4),
            "median_score": round(float(df['avg_qwen'].median()), 4),
            "good_rate": round(float((df['avg_qwen'] >= 0.7).mean()), 4),
            "bad_rate": round(float((df['avg_qwen'] <= 0.2).mean()), 4),
        },
        "ChatTS": {
            "mean_score": round(float(df['avg_chatts'].mean()), 4),
            "median_score": round(float(df['avg_chatts'].median()), 4),
            "good_rate": round(float((df['avg_chatts'] >= 0.7).mean()), 4),
            "bad_rate": round(float((df['avg_chatts'] <= 0.2).mean()), 4),
        },
        "Timer": {
            "mean_score": round(float(df['avg_timer'].mean()), 4),
            "median_score": round(float(df['avg_timer'].median()), 4),
            "good_rate": round(float((df['avg_timer'] >= 0.7).mean()), 4),
            "bad_rate": round(float((df['avg_timer'] <= 0.2).mean()), 4),
        },
    },
    "eval_categories": {k: int(v) for k, v in eval_dist.items()},
}

summary_path = os.path.join(OUTPUT_DIR, "scores_summary.json")
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n统计摘要已保存: {summary_path}")
print("\n✅ Excel 评分数据清洗完成！")
