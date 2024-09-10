#导入边
edges_csv_path = './Downloads/amazon_edge.csv' 
  
# 读取CSV文件并导入边数据  
def import_edges_from_csv(driver, csv_file_path):  
    with open(csv_file_path, mode='r', encoding='utf-8') as file:  
        reader = csv.DictReader(file)  
        with driver.session() as session:  
            for row in reader:  
                srcnodes = row['srcnodes']  
                dstnodes = row['dstnodes']  
                edgeId = row['edgeId'] 
                etype = row['etype']  
                # 使用Cypher查询创建关系  #根据自己的关联需求自定义
                session.run(  
                    "MATCH (a:nodeId {id: $srcnodes}), (b:nodeId {id: $dstnodes}) "  
                    "CREATE (a)-[:`RELTYPE` {etype: $etype}]->(b)",  
                    srcnodes=srcnodes,  
                    dstnodes=dstnodes, 
                    edgeId=edgeId,
                    etype=etype 
                )  
  
# 执行边导入  
import_edges_from_csv(driver, edges_csv_path)  
  
# 现在可以关闭驱动程序了  
driver.close()