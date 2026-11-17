from neo4j import GraphDatabase

NEO4J_URI = 'neo4j+s://d4296b49.databases.neo4j.io'
driver = GraphDatabase.driver(NEO4J_URI, auth=('d4296b49', 'JGdotqjbqK87YRGcnXtpoPsPqu2-iFf7h9lqsnGEG-Q'))

with driver.session(database='d4296b49') as session:
    print('--- 1. Single-hop lookups ---')
    print('Lexapro:', session.run("MATCH (d:Drug {name: 'Lexapro'})-[t:TREATS]->(c:Condition) RETURN c.name AS cond, t.avg_rating AS rating, t.review_count AS count").data())
    print('Ozempic:', session.run("MATCH (d:Drug {name: 'Ozempic'})-[t:TREATS]->(c:Condition) RETURN c.name AS cond, t.avg_rating AS rating, t.review_count AS count").data())
    print('Ambien:', session.run("MATCH (d:Drug {name: 'Ambien'})-[t:TREATS]->(c:Condition) RETURN c.name AS cond, t.avg_rating AS rating, t.review_count AS count").data())
    print('Vyvanse:', session.run("MATCH (d:Drug {name: 'Vyvanse'})-[t:TREATS]->(c:Condition) RETURN c.name AS cond, t.avg_rating AS rating, t.review_count AS count").data())
    print('Tramadol:', session.run("MATCH (d:Drug {name: 'Tramadol'})-[t:TREATS]->(c:Condition) RETURN c.name AS cond, t.avg_rating AS rating, t.review_count AS count").data())
    print('Accutane:', session.run("MATCH (d:Drug {name: 'Accutane'})-[t:TREATS]->(c:Condition) RETURN c.name AS cond, t.avg_rating AS rating, t.review_count AS count").data())

    print('\n--- 2. Multi-hop queries ---')
    print('Depression drugs causing insomnia:', session.run("MATCH (c:Condition {name: 'Depression'})<-[:TREATS]-(d:Drug)-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect {name: 'insomnia'}) RETURN d.name AS drug, r.report_count AS reports, r.frequency_pct AS freq ORDER BY r.report_count DESC LIMIT 5").data())
    print('Anxiety drugs causing nausea:', session.run("MATCH (c:Condition {name: 'Anxiety'})<-[:TREATS]-(d:Drug)-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect {name: 'nausea'}) RETURN d.name AS drug, r.report_count AS reports ORDER BY r.report_count DESC LIMIT 5").data())
    print('Drugs treating both Depression and Pain:', session.run("MATCH (d:Drug)-[:TREATS]->(c1:Condition {name: 'Depression'}), (d)-[:TREATS]->(c2:Condition {name: 'Pain'}) RETURN d.name AS drug, d.avg_rating AS rating LIMIT 5").data())
    print('Bipolar Disorder drugs causing tremor:', session.run("MATCH (c:Condition {name: 'Bipolar Disorder'})<-[:TREATS]-(d:Drug)-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect {name: 'tremor'}) RETURN d.name AS drug, r.report_count AS reports LIMIT 5").data())

    print('\n--- 3. Aggregation queries ---')
    print('Top rated drugs for Diabetes Type 2 (min 5 reviews):', session.run("MATCH (d:Drug)-[t:TREATS]->(c:Condition {name: 'Diabetes, Type 2'}) WHERE t.review_count >= 5 RETURN d.name AS drug, t.avg_rating AS rating, t.review_count AS count ORDER BY t.avg_rating DESC LIMIT 5").data())
    print('Top rated drugs for Acne (min 5 reviews):', session.run("MATCH (d:Drug)-[t:TREATS]->(c:Condition {name: 'Acne'}) WHERE t.review_count >= 5 RETURN d.name AS drug, t.avg_rating AS rating, t.review_count AS count ORDER BY t.avg_rating DESC LIMIT 5").data())
    print('Side effects of Lexapro:', session.run("MATCH (d:Drug {name: 'Lexapro'})-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect) RETURN s.name AS se, r.report_count AS count ORDER BY count DESC LIMIT 5").data())
    print('Total reviews and average rating for Pristiq vs Zoloft in Depression:', session.run("MATCH (d:Drug)-[t:TREATS]->(c:Condition {name: 'Depression'}) WHERE d.name IN ['Pristiq', 'Zoloft'] RETURN d.name AS drug, t.review_count AS count, t.avg_rating AS rating").data())

driver.close()
