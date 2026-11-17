"""
TASK 2: Graph Model Design & Load into Neo4j Aura
- Connects to Neo4j Aura using credentials
- Creates uniqueness constraints and full-text indexes
- Batched parameterized loading (UNWIND ... MERGE)
- Exploratory Cypher queries with results verification
"""

import os
import re
import json
import time
import pandas as pd
from neo4j import GraphDatabase

# Neo4j Aura Credentials
NEO4J_URI = "neo4j+s://d4296b49.databases.neo4j.io"
NEO4J_USERNAME = "d4296b49"
NEO4J_PASSWORD = "JGdotqjbqK87YRGcnXtpoPsPqu2-iFf7h9lqsnGEG-Q"
NEO4J_DATABASE = "d4296b49"

csv_path = r'data\drug_reviews_sample.csv'
df = pd.read_csv(csv_path)

SE_CAT_MAP = {
    'nausea': 'Gastrointestinal',
    'headache': 'Neurological',
    'dizziness': 'Neurological',
    'fatigue': 'General',
    'insomnia': 'Psychiatric',
    'drowsiness': 'General',
    'dry mouth': 'Gastrointestinal',
    'weight gain': 'Metabolic',
    'weight loss': 'Metabolic',
    'tremor': 'Neurological',
    'diarrhea': 'Gastrointestinal',
    'constipation': 'Gastrointestinal',
    'brain zaps': 'Neurological',
    'sweating': 'General',
    'loss of appetite': 'Metabolic',
    'blurred vision': 'Neurological',
    'stomach pain': 'Gastrointestinal',
    'vomiting': 'Gastrointestinal',
    'restlessness': 'Psychiatric',
    'palpitations': 'Cardiovascular',
    'agitation': 'Psychiatric',
    'nightmares': 'Psychiatric',
    'tinnitus': 'Neurological',
    'muscle pain': 'General',
    'sexual dysfunction': 'Endocrine',
    'hair loss': 'Dermatological',
    'brain fog': 'Cognitive'
}

print(f"Connecting to Neo4j Aura at {NEO4J_URI}...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
driver.verify_connectivity()
print("Connected successfully!")

t0 = time.time()

with driver.session(database=NEO4J_DATABASE) as session:
    print("Clearing any prior nodes in database...")
    session.run("MATCH (n) DETACH DELETE n")

    print("Creating uniqueness constraints...")
    constraints = [
        "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE;",
        "CREATE CONSTRAINT condition_name_unique IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE;",
        "CREATE CONSTRAINT sideeffect_name_unique IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE;",
        "CREATE CONSTRAINT review_id_unique IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE;",
        "CREATE CONSTRAINT patient_username_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.username IS UNIQUE;",
        "CREATE CONSTRAINT intake_duration_unique IF NOT EXISTS FOR (i:IntakeDuration) REQUIRE i.duration IS UNIQUE;"
    ]
    for c in constraints:
        try:
            session.run(c)
        except Exception as e:
            print(f"Constraint notice: {e}")

    print("Creating full-text indexes...")
    indexes = [
        "CREATE FULLTEXT INDEX drug_name_fulltext IF NOT EXISTS FOR (n:Drug) ON EACH [n.name];",
        "CREATE FULLTEXT INDEX condition_name_fulltext IF NOT EXISTS FOR (n:Condition) ON EACH [n.name];",
        "CREATE FULLTEXT INDEX sideeffect_name_fulltext IF NOT EXISTS FOR (n:SideEffect) ON EACH [n.name];",
        "CREATE FULLTEXT INDEX review_text_fulltext IF NOT EXISTS FOR (n:Review) ON EACH [n.text];"
    ]
    for idx in indexes:
        try:
            session.run(idx)
        except Exception as e:
            print(f"Index notice: {e}")

    # 1. Load Conditions
    print("Loading Condition nodes...")
    cond_stats = df.groupby('Condition').agg(
        review_count=('ReviewId', 'count'),
        avg_rating=('Rating', 'mean'),
        drug_count=('MedicineName', 'nunique')
    ).reset_index().to_dict('records')

    session.run("""
    UNWIND $batch AS row
    MERGE (c:Condition {name: row.Condition})
    SET c.review_count = row.review_count,
        c.avg_rating = round(row.avg_rating, 2),
        c.drug_count = row.drug_count
    """, batch=cond_stats)

    # 2. Load Drugs
    print("Loading Drug nodes...")
    drug_stats = df.groupby('MedicineName').agg(
        review_count=('ReviewId', 'count'),
        avg_rating=('Rating', 'mean'),
        avg_likes=('NumberOfLikes', 'mean')
    ).reset_index().to_dict('records')

    session.run("""
    UNWIND $batch AS row
    MERGE (d:Drug {name: row.MedicineName})
    SET d.review_count = row.review_count,
        d.avg_rating = round(row.avg_rating, 2),
        d.avg_likes = round(row.avg_likes, 2)
    """, batch=drug_stats)

    # 3. Load SideEffects
    print("Loading SideEffect nodes...")
    se_records = [{'name': name, 'category': cat} for name, cat in SE_CAT_MAP.items()]
    session.run("""
    UNWIND $batch AS row
    MERGE (s:SideEffect {name: row.name})
    SET s.category = row.category
    """, batch=se_records)

    # 4. Load IntakeDurations
    print("Loading IntakeDuration nodes...")
    intakes = [{'duration': d} for d in df['IntakeDuration'].dropna().unique()]
    session.run("""
    UNWIND $batch AS row
    MERGE (i:IntakeDuration {duration: row.duration})
    """, batch=intakes)

    # 5. Load Patients (Batch in chunks)
    print("Loading Patient nodes...")
    unique_patients = [{'username': u} for u in df['UserName'].dropna().unique()]
    BATCH_SIZE = 500
    for i in range(0, len(unique_patients), BATCH_SIZE):
        batch = unique_patients[i:i + BATCH_SIZE]
        session.run("""
        UNWIND $batch AS row
        MERGE (p:Patient {username: row.username})
        """, batch=batch)

    # 6. Load Reviews and Connect to Drug, Condition, Patient, IntakeDuration
    print("Loading Review nodes and basic relationships...")
    reviews_data = []
    for _, row in df.iterrows():
        reviews_data.append({
            'id': row['ReviewId'],
            'rating': float(row['Rating']),
            'date': str(row['ReviewDate']),
            'likes': int(row['NumberOfLikes']),
            'length': int(row['ReviewLength']),
            'text': str(row['Reviews']),
            'drug': str(row['MedicineName']),
            'condition': str(row['Condition']),
            'username': str(row['UserName']),
            'intake': str(row['IntakeDuration']),
            'side_effects': [s.strip() for s in row['SideEffects_Str'].split('; ')] if row['SideEffects_Str'] != 'None Reported' else []
        })

    for i in range(0, len(reviews_data), BATCH_SIZE):
        batch = reviews_data[i:i + BATCH_SIZE]
        session.run("""
        UNWIND $batch AS row
        MERGE (r:Review {id: row.id})
        SET r.rating = row.rating,
            r.date = row.date,
            r.likes = row.likes,
            r.length = row.length,
            r.text = row.text

        WITH r, row
        MATCH (d:Drug {name: row.drug})
        MERGE (r)-[:REVIEWS_DRUG]->(d)

        WITH r, row
        MATCH (c:Condition {name: row.condition})
        MERGE (r)-[:FOR_CONDITION]->(c)

        WITH r, row
        MATCH (p:Patient {username: row.username})
        MERGE (p)-[:POSTED]->(r)

        WITH r, row
        MATCH (i:IntakeDuration {duration: row.intake})
        MERGE (r)-[:HAS_INTAKE]->(i)

        WITH r, row
        UNWIND row.side_effects AS se_name
        MATCH (s:SideEffect {name: se_name})
        MERGE (r)-[:REPORTS_SIDE_EFFECT]->(s)
        """, batch=batch)

    # 7. Create Direct Aggregated Relationships (Drug)-[:TREATS]->(Condition)
    print("Creating aggregated (Drug)-[:TREATS]->(Condition) relationships...")
    treats_data = df.groupby(['MedicineName', 'Condition']).agg(
        review_count=('ReviewId', 'count'),
        avg_rating=('Rating', 'mean')
    ).reset_index().to_dict('records')

    session.run("""
    UNWIND $batch AS row
    MATCH (d:Drug {name: row.MedicineName})
    MATCH (c:Condition {name: row.Condition})
    MERGE (d)-[t:TREATS]->(c)
    SET t.review_count = row.review_count,
        t.avg_rating = round(row.avg_rating, 2)
    """, batch=treats_data)

    # 8. Create Direct Aggregated Relationships (Drug)-[:CAUSES_SIDE_EFFECT]->(SideEffect)
    print("Creating aggregated (Drug)-[:CAUSES_SIDE_EFFECT]->(SideEffect) relationships...")
    drug_se_records = []
    for drug_name, group in df.groupby('MedicineName'):
        total_revs = len(group)
        se_counter = {}
        for se_str in group['SideEffects_Str']:
            if se_str != 'None Reported':
                for s in se_str.split('; '):
                    se_counter[s] = se_counter.get(s, 0) + 1
        for s, count in se_counter.items():
            freq_pct = round((count / total_revs) * 100, 2)
            drug_se_records.append({
                'drug': drug_name,
                'side_effect': s,
                'count': count,
                'freq_pct': freq_pct
            })

    for i in range(0, len(drug_se_records), BATCH_SIZE):
        batch = drug_se_records[i:i + BATCH_SIZE]
        session.run("""
        UNWIND $batch AS row
        MATCH (d:Drug {name: row.drug})
        MATCH (s:SideEffect {name: row.side_effect})
        MERGE (d)-[r:CAUSES_SIDE_EFFECT]->(s)
        SET r.report_count = row.count,
            r.frequency_pct = row.freq_pct
        """, batch=batch)

    load_time = time.time() - t0
    print(f"\nGraph successfully loaded into Neo4j Aura in {load_time:.2f} seconds!")

    # Verify counts
    print("\n--- NODE COUNTS ---")
    node_counts = session.run("""
    MATCH (n) RETURN head(labels(n)) AS label, count(n) AS count ORDER BY count DESC
    """).data()
    for row in node_counts:
        print(f"  {row['label']}: {row['count']}")

    print("\n--- RELATIONSHIP COUNTS ---")
    rel_counts = session.run("""
    MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC
    """).data()
    for row in rel_counts:
        print(f"  {row['rel_type']}: {row['count']}")

    # 5 Exploratory Cypher Queries per Task 2
    print("\n" + "="*60)
    print("TASK 2: 5 EXPLORATORY CYPHER QUERIES WITH OUTPUT")
    print("="*60)

    queries = [
        (
            "Query 1: Top 5 Highest Rated Drugs for Anxiety (min 10 reviews)",
            """
            MATCH (d:Drug)-[t:TREATS]->(c:Condition {name: 'Anxiety'})
            WHERE t.review_count >= 10
            RETURN d.name AS Drug, t.avg_rating AS AvgRating, t.review_count AS Reviews
            ORDER BY t.avg_rating DESC, Reviews DESC LIMIT 5
            """
        ),
        (
            "Query 2: Multi-hop: Drugs treating Depression that also cause Insomnia",
            """
            MATCH (c:Condition {name: 'Depression'})<-[:TREATS]-(d:Drug)-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect {name: 'insomnia'})
            RETURN d.name AS Drug, r.report_count AS Reports, r.frequency_pct AS FreqPct
            ORDER BY r.report_count DESC LIMIT 5
            """
        ),
        (
            "Query 3: Shared Medications: Drugs prescribed for both Depression and Anxiety",
            """
            MATCH (d:Drug)-[:TREATS]->(c1:Condition {name: 'Depression'})
            MATCH (d)-[:TREATS]->(c2:Condition {name: 'Anxiety'})
            RETURN d.name AS SharedDrug, d.avg_rating AS OverallRating, d.review_count AS TotalReviews
            ORDER BY d.review_count DESC LIMIT 5
            """
        ),
        (
            "Query 4: Aggregation: Most Frequently Reported Side Effects across all drugs",
            """
            MATCH (s:SideEffect)<-[r:CAUSES_SIDE_EFFECT]-(d:Drug)
            RETURN s.name AS SideEffect, s.category AS System, sum(r.report_count) AS TotalMentions, count(DISTINCT d) AS DrugCount
            ORDER BY TotalMentions DESC LIMIT 5
            """
        ),
        (
            "Query 5: Complex Multi-hop: Alternative drugs for Type 2 Diabetes with lowest nausea reports",
            """
            MATCH (d:Drug)-[:TREATS]->(c:Condition {name: 'Diabetes, Type 2'})
            OPTIONAL MATCH (d)-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect {name: 'nausea'})
            RETURN d.name AS Drug, d.avg_rating AS Rating, coalesce(r.report_count, 0) AS NauseaMentions
            ORDER BY Rating DESC, NauseaMentions ASC LIMIT 5
            """
        )
    ]

    for title, q in queries:
        print(f"\n{title}:")
        results = session.run(q).data()
        for res in results:
            print("  ", res)

driver.close()
