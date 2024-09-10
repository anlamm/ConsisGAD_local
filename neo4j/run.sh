#!/bin/bash

# python -u qiushi.py --dataset amazon 2>&1 | tee cypher.log
# python -u qiushi.py --dataset yelp 2>&1 | tee -a cypher.log
python -u qiushi.py --dataset tfinance 2>&1 | tee -a cypher.log
# python -u createDB.py 2>&1 | tee -a cypher2.log
# python -u qiushi.parallel.py 2>&1 | tee cypherp.log

# neo4j-admin import --database=neo4j --nodes=amazon_node_100.2.csv --relationships=amazon_edge_100.2.csv

# LOAD CSV WITH HEADERS FROM 'file:///amazon_node_100.csv'  AS row MERGE(n:User {id: row.nodeId, label: row.label});