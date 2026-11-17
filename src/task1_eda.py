"""
TASK 1: Problem & Data Understanding
- Data Card generation (saved to data/data_card.json)
- Exploratory Data Analysis & Visualizations (saved to outputs/task1_eda.png)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

sample_path = r'data\drug_reviews_sample.csv'
df = pd.read_csv(sample_path)

# 1. Generate Data Card JSON
data_card = {
    "dataset_title": "DrugRev: A comprehensive customers reviews on drugs purchasing and satisfaction",
    "source": "Mendeley Data",
    "mendeley_url": "https://data.mendeley.com/datasets/64cc5w5dxy/1",
    "doi": "10.17632/64cc5w5dxy.1",
    "published_date": "August 13, 2023",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "authors": "Muhammad Rizwan et al.",
    "domain": "Clinical Pharmacology, Pharmacovigilance, and Patient Satisfaction Analysis",
    "target_users": "Clinical researchers, pharmacists, prescribing physicians, and healthcare consumers",
    "raw_dataset_size": {
        "rows": 392510,
        "columns": 9,
        "file_size_mb": 162.56,
        "unique_drugs": 4715,
        "unique_conditions": 7452
    },
    "processed_sample_size": {
        "rows": len(df),
        "columns": len(df.columns),
        "unique_drugs": int(df['MedicineName'].nunique()),
        "unique_conditions": int(df['Condition'].nunique()),
        "reviews_with_adrs": int((df['SideEffects_Str'] != 'None Reported').sum())
    },
    "schema_fields": [
        {"name": "ReviewId", "type": "string", "description": "Unique review identifier (REV_00001 to REV_04500)"},
        {"name": "MedicineName", "type": "string", "description": "Brand or generic drug name evaluated"},
        {"name": "Condition", "type": "string", "description": "Clinical diagnosis / therapeutic indication"},
        {"name": "Rating", "type": "float", "description": "Quantitative customer satisfaction rating (1.0 to 10.0)"},
        {"name": "ReviewDate", "type": "string", "description": "Timestamp of review publication"},
        {"name": "UserName", "type": "string", "description": "Patient screen name or pseudonym"},
        {"name": "IntakeDuration", "type": "string", "description": "Categorized duration of drug intake"},
        {"name": "ReviewLength", "type": "integer", "description": "Length of customer review in characters"},
        {"name": "NumberOfLikes", "type": "integer", "description": "Community helpfulness upvotes count"},
        {"name": "SideEffects_Str", "type": "string", "description": "Validated extracted adverse drug reactions"},
        {"name": "Reviews", "type": "string", "description": "Full narrative text of patient experience"}
    ],
    "quality_issues_and_remediation": [
        {"issue": "Whitespace in Drug Names", "fix": "Trimmed leading and trailing whitespace across all drug strings"},
        {"issue": "Prefix in Conditions", "fix": "Removed 'For ' prefix (e.g. 'For Anxiety' -> 'Anxiety')"},
        {"issue": "Missing User Names", "fix": "Imputed 56 null entries with 'Anonymous_User'"},
        {"issue": "Malformed Intake Times", "fix": "Standardized non-duration string entries to 'Not Specified'"},
        {"issue": "Unstructured Adverse Reactions", "fix": "Applied clinical rule-based keyword extraction for 27 standardized ADRs"}
    ],
    "sampling_strategy": "Stratified balanced sampling across top 10 therapeutic conditions ensuring >=30 reviews per drug-condition pair for rich multi-hop graph connectivity while adhering to Neo4j Aura Free tier node limits."
}

with open(r'data\data_card.json', 'w', encoding='utf-8') as f:
    json.dump(data_card, f, indent=2)
print("Data Card saved to data/data_card.json")

# 2. Generate Exploratory Data Analysis Figures
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("DrugRev: Exploratory Data Analysis & Pharmacovigilance Distributions", fontsize=16, fontweight='bold', y=0.98)

# Subplot 1: Rating Distribution
sns.histplot(df['Rating'], bins=10, kde=True, ax=axes[0, 0], color='#2b5c8f', edgecolor='black')
axes[0, 0].set_title("Customer Rating Distribution (1-10)", fontweight='bold')
axes[0, 0].set_xlabel("Rating")
axes[0, 0].set_ylabel("Review Count")

# Subplot 2: Review Length Distribution (log scale)
sns.histplot(df['ReviewLength'], bins=40, ax=axes[0, 1], color='#e07a5f', kde=True)
axes[0, 1].set_title("Review Character Length Distribution", fontweight='bold')
axes[0, 1].set_xlabel("Review Length (Characters)")
axes[0, 1].set_ylabel("Count")

# Subplot 3: Top Conditions
cond_counts = df['Condition'].value_counts()
sns.barplot(y=cond_counts.index, x=cond_counts.values, ax=axes[0, 2], palette="crest")
axes[0, 2].set_title("Balanced Reviews Across 10 Clinical Conditions", fontweight='bold')
axes[0, 2].set_xlabel("Review Count")

# Subplot 4: Top 10 Drugs by Review Count
top_drugs = df['MedicineName'].value_counts().head(10)
sns.barplot(y=top_drugs.index, x=top_drugs.values, ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title("Top 10 Evaluated Medications in Sample", fontweight='bold')
axes[1, 0].set_xlabel("Review Count")

# Subplot 5: Top 10 Extracted Side Effects
from collections import Counter
all_se = []
for se_str in df['SideEffects_Str']:
    if se_str != 'None Reported':
        for s in se_str.split('; '):
            all_se.append(s.title())
se_counts = Counter(all_se).most_common(10)
se_df = pd.DataFrame(se_counts, columns=['SideEffect', 'Count'])
sns.barplot(y=se_df['SideEffect'], x=se_df['Count'], ax=axes[1, 1], palette="flare")
axes[1, 1].set_title("Top 10 Reported Adverse Drug Reactions (ADRs)", fontweight='bold')
axes[1, 1].set_xlabel("Occurrence Count")

# Subplot 6: Intake Duration Breakdown
intake_counts = df['IntakeDuration'].value_counts()
sns.barplot(y=intake_counts.index, x=intake_counts.values, ax=axes[1, 2], palette="magma")
axes[1, 2].set_title("Patient Intake Duration Breakdown", fontweight='bold')
axes[1, 2].set_xlabel("Patient Count")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
os.makedirs("outputs", exist_ok=True)
out_png = r'outputs\task1_eda.png'
plt.savefig(out_png, dpi=300)
plt.close()
print(f"EDA plot saved successfully to {out_png}")
