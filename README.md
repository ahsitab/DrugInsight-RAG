# DrugInsight-RAG: GraphRAG vs Vector RAG for Clinical Pharmacovigilance

**Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh**  
**Course:** AIML506 — Machine Learning Systems (Fall 2026)  
**Course Instructor:** Dr. Mohammad Rifat Ahmmad Rashid  
**Authors:**
- **Asfar Hossain Sitab** (Student ID: `2026-2-74-008`)
- **Parmita Hossain Simia** (Student ID: `2026-2-74-009`)

🔗 **Public Kaggle Notebook:** [https://www.kaggle.com/code/asfarhossainsitab/aiml-506-project-drugreview](https://www.kaggle.com/code/asfarhossainsitab/aiml-506-project-drugreview)  
🔗 **GitHub Repository:** [https://github.com/ahsitab/DrugInsight-RAG](https://github.com/ahsitab/DrugInsight-RAG)  
📄 **Academic Report (PDF):** [`AIML506_GraphRAG_vs_VectorRAG.pdf`](AIML506_GraphRAG_vs_VectorRAG.pdf) (15-Page IEEE Conference Paper)

---

## 📌 Executive Summary

This study presents **DrugInsight-RAG**, a rigorous empirical comparative evaluation of a Knowledge Graph-based Retrieval-Augmented Generation system (**GraphRAG** implemented via **Neo4j Aura**) and a Dense Vector-based Retrieval-Augmented Generation system (**Vector RAG** implemented via **ChromaDB** with `all-MiniLM-L6-v2`) on real-world patient-authored drug reviews from the Mendeley Data *DrugRev* corpus ([DOI: 10.17632/64cc5w5dxy.1](https://doi.org/10.17632/64cc5w5dxy.1)).

Both systems operate on an identical, standardized common evidence corpus of **5,584 records** derived from a balanced stratified sample of **4,500 reviews** across 10 clinical conditions. Evaluation is performed across 30 questions (6 each of 5 query archetypes from Table 1 of the course rubric) using **Mistral 7B Instruct v0.3** (4-bit NF4) as the primary generator on a Tesla T4 GPU, with **Qwen 2.5 7B Instruct** as an automated LLM judge.

---

## 📊 Core Empirical Findings (Table V)

The table below records the complete evaluation metrics extracted directly from the executed Kaggle notebook run:

| Evaluation Metric | GraphRAG (Neo4j Aura) | Vector RAG (ChromaDB) | System Advantage / Findings |
| :--- | :---: | :---: | :--- |
| **Hit@5 (Retrieval Recall)** | **1.000** | 0.333 | **GraphRAG (+0.667)** — Graph retrieved gold evidence in 100% of answerable items |
| **Recall@5** | **1.000** | 0.333 | **GraphRAG (+0.667)** |
| **Precision@5** | **0.333** | 0.067 | **GraphRAG (+0.266)** |
| **Mean Reciprocal Rank (MRR)** | **0.967** | 0.130 | **GraphRAG (+0.837)** — Gold evidence placed at rank 1 in almost all queries |
| **nDCG@5** | **0.973** | 0.181 | **GraphRAG (+0.792)** |
| **Key-Item Accuracy** | **0.667** | 0.633 | **GraphRAG (+0.034)** |
| **Token F1 (Word Overlap)** | **0.410** | 0.366 | **GraphRAG (+0.044)** |
| **LLM-Judge Correctness (Qwen 2.5 7B)** | 0.642 | **0.667** | **Vector RAG (+0.025)** |
| **Groundedness (LLM Judge)** | 0.708 | **0.717** | **Vector RAG (+0.009)** |
| **Citation Validity** | **0.833** | 0.250 | **GraphRAG (+0.583)** — Verified bracketed evidence ID citations |
| **Out-of-Scope Refusal (6 items)** | **1.000** | **1.000** | **Tied** — Controlled keyword refusal gate |
| **Graph Entity-Link F1** | **1.000** | N/A | **GraphRAG** |
| **Median Total Latency (s)** | 8.948 s | **7.069 s** | **Vector RAG (1.88 s faster end-to-end)** |
| **Median Retrieval Latency (s)** | 0.464 s | **0.060 s** | **Vector RAG (0.40 s faster retrieval)** |
| **Median Generation Latency (s)** | 8.392 s | **6.996 s** | **Vector RAG (1.40 s faster generation)** |
| **Mean Evidence Context (Tokens)** | **385.8 tokens** | 463.3 tokens | **GraphRAG (-77.5 tokens/query context reduction)** |

---

## 🎯 Retrieval Performance by Question Category (Table VI)

Hit@5 retrieval performance broken down by question archetype ($n=6$ per category):

| Question Archetype | GraphRAG Hit@5 | Vector RAG Hit@5 | Key Diagnostic Observation |
| :--- | :---: | :---: | :--- |
| **Single-Hop** | **1.000** | 0.833 | Both systems perform well on direct drug-condition factual lookups. |
| **Multi-Hop** | **1.000** | **0.000** | **Vector RAG multi-hop path blindness:** Dense embeddings cannot follow relational joins across condition $\rightarrow$ drug $\rightarrow$ adverse reaction. |
| **Aggregation** | **1.000** | 0.333 | **Vector RAG chunking blindness:** Dense search cannot execute global set grouping or ranking across hundreds of medications. |
| **Semantic / Descriptive** | **1.000** | 0.167 | Graph traversal pinpoints structured review summaries with highest community upvotes. |
| **Unanswerable** | N/A | N/A | Both systems correctly routed to out-of-scope refusal (100% refusal accuracy). |

---

## 🤖 4-Model Family Ablation Study (Table VII)

Cross-category evaluation of 5 benchmark queries across four distinct open-source LLM architectures:

| Model Architecture | Parameter Size | Retrieval Route | Key-Item Acc. | Token F1 | Median Gen. Time (s) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Mistral 7B Instruct v0.3** | 7.3B | GraphRAG | 0.60 | **0.437** | 8.70 s |
| **Mistral 7B Instruct v0.3** | 7.3B | Vector RAG | 0.60 | 0.345 | 7.19 s |
| **Qwen 2.5 7B Instruct** | 7.0B | GraphRAG | 0.60 | **0.396** | 8.88 s |
| **Qwen 2.5 7B Instruct** | 7.0B | Vector RAG | 0.60 | 0.329 | **5.43 s** |
| **Zephyr 7B Beta** | 7.0B | GraphRAG | 0.60 | **0.447** | 12.19 s |
| **Zephyr 7B Beta** | 7.0B | Vector RAG | 0.60 | 0.328 | 12.05 s |
| **Phi-3.5 Mini Instruct** | 3.8B | GraphRAG | 0.60 | **0.424** | 14.23 s |
| **Phi-3.5 Mini Instruct** | 3.8B | Vector RAG | 0.60 | 0.358 | 9.78 s |

---

## 🔍 Forensic Root Cause Failure Analysis (Table VIII)

| ID | System | Query Type | Observed Symptom | Root Cause Diagnosis |
| :--- | :--- | :--- | :--- | :--- |
| **AG1** | VectorRAG | Aggregation | Returned Toprol-XL and Bystolic instead of Azor, Atenolol, Diovan. | **Vector Chunking Blindness:** Vector similarity cannot rank top-3 by rating with $\ge 5$ review filter across all condition drugs. |
| **AG4** | VectorRAG | Aggregation | Answered Orlistat (7 reviews) instead of Phentermine (100 reviews). | **Global Maximum Miss:** Dense retrieval matched a local document mention rather than the corpus global maximum. |
| **AG2** | VectorRAG | Aggregation | Answered agitation (16 reviews) instead of insomnia (223) and nausea (187). | **Missing Global Frequency Leaders:** Local nearest-neighbor search returned a specific psychiatric symptom rather than corpus-wide frequency counts. |
| **MH6** | GraphRAG | Multi-hop | Retrieved gold evidence (Recall 1.0), but generator omitted required items. | **Generation Output Bottleneck:** The 160-token output limit caused the model to truncate the complete medication list. |
| **AG3** | VectorRAG | Aggregation | Answered Pain (450 reviews), scored wrong against ADHD (450 reviews). | **Benchmark Tie Defect:** All 10 conditions had exactly 450 reviews due to stratified sampling; both Pain and ADHD were tied for the maximum. |

---

## 🏗️ System Architecture & Graph Topology

### Knowledge Graph (Neo4j Aura Cloud — `Instance02`)
- **Scoped Project ID:** `AIML506_DRUGREV_GRAPHRAG_VECTORRAG_V3`
- **Total Nodes:** **8,411**
  - `Review`: 4,500
  - `Patient`: 3,395
  - `Drug`: 470
  - `SideEffect`: 27
  - `Condition`: 10
  - `IntakeDuration`: 9
- **Total Relationships:** **20,599**
  - `POSTED`: 4,500
  - `REVIEWS_DRUG`: 4,500
  - `FOR_CONDITION`: 4,500
  - `HAS_INTAKE`: 4,500
  - `REPORTS_SIDE_EFFECT`: 1,317
  - `TREATS`: 568
  - `CAUSES_SIDE_EFFECT`: 714 (lexical co-mention association)
- **Graph Rebuild Time:** 13.29 seconds

### Vector Database (ChromaDB)
- **Embedding Model:** `all-MiniLM-L6-v2` (384-dimensional dense vectors, cosine metric)
- **Indexed Corpus:** **5,584 documents**
  - 4,500 Raw Patient Review Narratives
  - 568 Medication–Condition Summaries
  - 470 Medication Summaries
  - 27 Matched-Term (ADR) Summaries
  - 10 Condition Summaries
  - 9 Intake Duration Summaries
- **Indexing Time:** 299.07 seconds

---

## 📁 Repository Structure

```
DrugInsight-RAG/
├── README.md                                    # Comprehensive project documentation & results
├── .gitignore                                   # Excludes large raw datasets (>100MB)
├── .gitattributes                               # Binary protection for PDFs & images
├── AIML506_GraphRAG_vs_VectorRAG.pdf            # 15-Page IEEE Conference Paper
├── DrugInsight-RAG_Report.pdf                   # Publication Report (copy)
├── data/
│   ├── drug_reviews_sample.csv                  # Stratified 4,500 review sample (1.82 MB)
│   ├── benchmark.json                           # 30 Evaluation questions with gold references
│   └── data_card.json                           # Dataset provenance & quality metadata
├── notebooks/
│   └── aiml-506-project-drugreview.ipynb        # Complete executed Kaggle notebook (68 cells, 123 outputs)
└── report/
    ├── AIML506_GraphRAG_vs_VectorRAG.pdf        # IEEE Conference Report
    ├── AIML506_2026-2-74-008_GraphRAG_vs_VectorRAG.pdf # Named with Student ID
    └── AIML506_Project_Report.tex               # Overleaf-ready LaTeX source
```

---

## 🔬 Dataset & Ethical Considerations
- **Dataset:** *DrugRev: A comprehensive customers reviews on drugs purchasing and satisfaction*, Mendeley Data, Version 1, DOI: [10.17632/64cc5w5dxy.1](https://doi.org/10.17632/64cc5w5dxy.1).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Clinical Notice:** Patient reviews reflect subjective experiences, not verified randomized controlled clinical trials. Adverse reaction extractions are lexical co-mentions and do not establish pharmacological causality. The system is designed for exploratory research and pharmacovigilance intelligence only.

---

## 👥 Authors
- **Asfar Hossain Sitab** — Student ID: `2026-2-74-008` — Department of CSE, East West University, Dhaka, Bangladesh
- **Parmita Hossain Simia** — Student ID: `2026-2-74-009` — Department of CSE, East West University, Dhaka, Bangladesh
- **Course:** AIML506 — Machine Learning Systems (Instructor: Dr. Mohammad Rifat Ahmmad Rashid)
