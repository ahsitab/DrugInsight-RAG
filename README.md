# DrugInsight-RAG: GraphRAG vs Vector RAG for Clinical Pharmacovigilance

**Department of Computer Science and Engineering, East West University**  
**Course:** AIML506 — Machine Learning Systems  
**Course Instructor:** Dr. Mohammad Rifat Ahmmad Rashid  
**Author:** A. H. Sitab  
**Project Repository:** [DrugInsight-RAG (GitHub)](https://github.com/ahsitab/DrugInsight-RAG)  
**Dataset:** [DrugRev: Mendeley Data (DOI: 10.17632/64cc5w5dxy.1)](https://doi.org/10.17632/64cc5w5dxy.1)  

---

## 📌 Deliverables Summary
This project repository provides the complete implementation, evaluation engine, reproducible notebook, and IEEE conference paper:

1. **Academic Report (PDF & LaTeX):**
   - **Compiled IEEE Conference PDF:** [`AIML506_GraphRAG_vs_VectorRAG.pdf`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/AIML506_GraphRAG_vs_VectorRAG.pdf) (8 pages, fully illustrated, matches IEEE formatting).
   - **Overleaf-Ready LaTeX Source:** [`report/AIML506_Project_Report.tex`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/report/AIML506_Project_Report.tex).
2. **Public Kaggle Notebook:**
   - [`notebooks/AIML506_GraphRAG_vs_VectorRAG.ipynb`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/notebooks/AIML506_GraphRAG_vs_VectorRAG.ipynb).
   - Fully self-contained, runs top-to-bottom, reads credentials from Kaggle Secrets, and exports result tables to `/kaggle/working`.
3. **Live Neo4j Aura Cloud Knowledge Graph:**
   - **8,296 nodes** and **20,359 relationships** loaded and active on Neo4j Aura cloud (`Instance02`).
4. **Empirical Benchmark & Evaluation Results:**
   - 30-Question Benchmark with Gold Answers: [`outputs/benchmark.csv`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/benchmark.csv).
   - Overall 12-Metric Comparison: [`outputs/overall_results.csv`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/overall_results.csv).
   - Results by Question Category: [`outputs/results_by_type.csv`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/results_by_type.csv).
   - 5-Model Ablation Study: [`outputs/ablation_results.csv`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/ablation_results.csv).
   - 5 Root-Cause Failure Analyses: [`outputs/failure_analysis.csv`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/failure_analysis.csv).
   - Build Cost & Efficiency: [`outputs/cost_comparison.csv`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/cost_comparison.csv).
   - Figures & Plots: [`outputs/task1_eda.png`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/task1_eda.png) and [`outputs/comparison_charts.png`](file:///e:/SITAB_MSc/AIML-506/Project_Drug_Review/outputs/comparison_charts.png).

---

## 📊 Core Empirical Findings

| Evaluation Metric | GraphRAG (Neo4j Aura) | Vector RAG (ChromaDB) | System Advantage |
|:---|:---:|:---:|:---|
| **Accuracy (Key-Item Match)** | **93.3%** | 66.7% | **GraphRAG (+26.6%)** |
| **Token F1 (SQuAD Overlap)** | **0.317** | 0.136 | **GraphRAG (+133% overlap)** |
| **Refusal Accuracy (Unanswerable)** | **100.0%** | 0.0% | **GraphRAG (Zero hallucination)** |
| **Mean Context Size (Tokens/query)** | **164.3 tokens** | 340.9 tokens | **GraphRAG (51.8% token savings)** |
| **Aggregation Queries Token Overhead**| **30.3 tokens** | 303.2 tokens | **GraphRAG (90.0% token reduction)** |
| **Median Retrieval Latency** | **0.210s** | 0.281s | **GraphRAG** |
| **Mean Reciprocal Rank (MRR)** | 0.675 | **0.797** | **Vector RAG (Broad matching)** |

---

## 🔬 Dataset & Domain Understanding (Task 1)
- **Title:** *DrugRev: A comprehensive customers reviews on drugs purchasing and satisfaction*
- **Source:** Mendeley Data ([https://data.mendeley.com/datasets/64cc5w5dxy/1](https://data.mendeley.com/datasets/64cc5w5dxy/1))
- **DOI:** `10.17632/64cc5w5dxy.1`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Published:** August 13, 2023
- **Raw Volume:** 392,510 patient evaluations across 4,715 drugs and 7,452 conditions (162.56 MB).
- **Quality Remediation:**
  1. Trimmed drug string whitespace.
  2. Normalized condition strings (stripped `For ` prefixes).
  3. Imputed missing usernames with `Anonymous_User`.
  4. Standardized malformed date-based intake times into `Not Specified`.
  5. Extracted 27 clinical adverse reactions (ADRs) mapped to physiological systems using boundary-delimited regex.
  6. Stratified 4,500 balanced reviews across 10 top conditions ensuring $\ge 30$ reviews per drug-condition pair.

---

## 🕸️ Property Graph Model Design (Task 2)
### Graph Schema Topologies:
- **`(:Drug)`** (295 nodes): `name`, `avg_rating`, `review_count`, `avg_likes`
- **`(:Condition)`** (10 nodes): `name`, `review_count`, `drug_count`
- **`(:SideEffect)`** (27 nodes): `name`, `category` (Gastrointestinal, Neurological, Psychiatric, Metabolic, etc.)
- **`(:Review)`** (4,500 nodes): `id`, `rating`, `date`, `likes`, `length`, `text`
- **`(:Patient)`** (3,456 nodes): `username`
- **`(:IntakeDuration)`** (8 nodes): `duration`

### Relationships (20,359 edges):
- `(:Drug)-[:TREATS]->(:Condition)` (353 edges)
- `(:Review)-[:REVIEWS_DRUG]->(:Drug)` (4,500 edges)
- `(:Review)-[:FOR_CONDITION]->(:Condition)` (4,500 edges)
- `(:Patient)-[:POSTED]->(:Review)` (4,500 edges)
- `(:Review)-[:HAS_INTAKE]->(:IntakeDuration)` (4,500 edges)
- `(:Review)-[:REPORTS_SIDE_EFFECT]->(:SideEffect)` (1,301 edges)
- `(:Drug)-[:CAUSES_SIDE_EFFECT]->(:SideEffect)` (705 edges)

---

## 🚀 Live Viva Defense Guide (5 December 2026)

Be prepared to answer these core questions during your viva:

### 1. "Explain your graph model design and why it captures more than the raw table."
- *Answer:* "A naive table mapping would simply create isolated review rows. Our model constructs explicit **`Drug`**, **`Condition`**, and **`SideEffect`** entity nodes. Crucially, we extract standardized clinical ADRs from the unstructured narrative text, linking each review to the adverse effect while computing aggregated **`[:CAUSES_SIDE_EFFECT]`** and **`[:TREATS]`** edges directly on the drug nodes. This enables direct multi-hop graph traversals (e.g., *Condition $\leftarrow$ Treats $\leftarrow$ Drug $\rightarrow$ Causes $\rightarrow$ SideEffect*) in constant time without scanning thousands of text rows."

### 2. "Run a Cypher query on your live Aura graph and explain the result."
Open the Neo4j Aura console at [https://console.neo4j.io](https://console.neo4j.io) and run:
```cypher
MATCH (c:Condition {name: 'Depression'})<-[:TREATS]-(d:Drug)-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect {name: 'insomnia'})
RETURN d.name AS Drug, r.report_count AS Reports, r.frequency_pct AS FreqPct
ORDER BY r.report_count DESC LIMIT 5;
```
- *Explanation:* "This is a 2-hop traversal finding medications prescribed for Depression that patients report as causing insomnia. Mirtazapine has 20 reports (25.97%), Trazodone has 16 reports (29.63%), and Quetiapine has 14 reports (25.45%)."

### 3. "Why does Vector RAG fail on aggregation and unanswerable questions?"
- *Answer:* "Dense vector embeddings operate locally by finding text chunks with similar semantic meaning. For aggregations (like ranking the top 3 diabetes drugs), Vector RAG cannot group or sort across hundreds of drugs—it merely returns 5 random individual patient reviews. For unanswerable queries (like asking for molecular synthesis or FDA pediatric dosage), vector databases unconditionally return the top-5 closest text passages, forcing the LLM to hallucinate answers. GraphRAG maintains strict schema boundaries, immediately refusing out-of-domain queries with 100% accuracy."

---

## 🛠️ Step-by-Step Instructions to Upload to Kaggle

1. Go to [https://www.kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Select **File → Import Notebook** and upload `notebooks/AIML506_GraphRAG_vs_VectorRAG.ipynb`.
3. In the Kaggle Notebook sidebar, go to **Add-ons → Secrets** and add:
   - `NEO4J_URI`: `neo4j+s://d4296b49.databases.neo4j.io`
   - `NEO4J_USERNAME`: `d4296b49`
   - `NEO4J_PASSWORD`: `JGdotqjbqK87YRGcnXtpoPsPqu2-iFf7h9lqsnGEG-Q`
   - `NEO4J_DATABASE`: `d4296b49`
   - (Optional) `GROQ_API_KEY`: for high-speed cloud generation.
4. Attach the dataset: Click **Add Input**, search for or upload `DrugReviews.csv` (or the processed `data/drug_reviews_sample.csv`).
5. Click **Run All** (Verify all cells execute cleanly with outputs saved).
6. Set notebook visibility to **Public** and copy the URL into your report title page.
