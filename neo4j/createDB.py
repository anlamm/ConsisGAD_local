import csv  
from neo4j import GraphDatabase  
from tqdm import tqdm

### Get number of lines of a file
### https://stackoverflow.com/questions/845058/how-to-get-the-line-count-of-a-large-file-cheaply-in-python
def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count
  
# 连接到Neo4j数据库  
uri = "bolt://localhost:7687"  
user = "neo4j"    #neo4j用户名
password = "asdf123456789"   #neo4j密码
driver = GraphDatabase.driver(uri, auth=(user, password))  
database = 'test'

with driver.session(database=database) as session:  
    session.run(  
        "MATCH (n) DETACH DELETE n;"
    )

#导入节点
print(f"Import Nodes ...")
# CSV文件路径  
nodes_csv_path = 'file:////Users/liuyue/Downloads/amazon_node.csv'  
  
# 读取CSV文件并导入节点数据  
def import_nodes_from_csv(driver, csv_file_path):   
    with driver.session(database=database) as session:  
        cypher = "LOAD CSV WITH HEADERS FROM $csv_file_path AS row MERGE(n:User {id: row.nodeId, label: row.label});"
        ret = session.run(  
            cypher,
            csv_file_path=csv_file_path,
        )  
        print(ret.single())


# 执行节点导入  
import_nodes_from_csv(driver, nodes_csv_path)  

print(f"Creating Index on nodeid")
with driver.session(database=database) as session:  
    ret = session.run("DROP INDEX nodeid_index IF EXISTS")
    ret = session.run(  
        "CREATE INDEX nodeid_index FOR (n:User) ON (n.nodeid)"
    )
    # print(ret.single()[0])
    print(ret.single())
  
# 不要在这里关闭驱动程序，因为后面还要导入边
#导入边
print(f"Import Edges ...")
edges_csv_path = 'file:////Users/liuyue/Downloads/amazon_edge.csv' 
  
# 读取CSV文件并导入边数据  
def import_edges_from_csv(driver, csv_file_path):  
    with driver.session(database=database) as session:  
        cypher = "PROFILE LOAD CSV WITH HEADERS FROM $csv_file_path AS row MATCH (a:User {id: row.srcnodes}) MATCH (b:User {id: row.dstnodes}) MERGE (a)-[:RELATION]->(b);"
        ret = session.run(  
            cypher,  
            csv_file_path=csv_file_path,
        )  
        print(ret.single())
  
# 执行边导入  
import_edges_from_csv(driver, edges_csv_path)  
  
# 现在可以关闭驱动程序了  
driver.close()