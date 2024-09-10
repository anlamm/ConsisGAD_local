import csv  
from neo4j import GraphDatabase  
from tqdm import tqdm
import argparse

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


if __name__ == "__main__":
  
    # 连接到Neo4j数据库  
    uri = "bolt://localhost:7687"  
    user = "neo4j"    #neo4j用户名
    password = "asdf123456789"   #neo4j密码
    driver = GraphDatabase.driver(uri, auth=(user, password))  
    database = 'neo4j'


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon")
    args = parser.parse_args()
    print(args)
    # dataset = "amazon"
    dataset = args.dataset
    nhop=1
    nodes_csv_path = f'/home/yliumh/github/ConsisGAD/data/offline/addfailure/CSV/{dataset}_node_{nhop}hop.csv' 
    edges_csv_path = f'/home/yliumh/github/ConsisGAD/data/offline/addfailure/CSV/{dataset}_edge_{nhop}hop.csv' 

    with driver.session(database=database) as session:  
        # session.run(  
        #     "MATCH (n) DETACH DELETE n;"
        # )


        print(f"Creating constrain on nodeid")
        ret = session.run(f"DROP INDEX {dataset}_nodeid_index IF EXISTS") 
        ret = session.run(f"DROP CONSTRAINT {dataset}_nodeid_contraint1 IF EXISTS")
        ret = session.run(f"DROP CONSTRAINT {dataset}_nodeid_contraint2 IF EXISTS")
        ret = session.run(  
            f"CREATE CONSTRAINT {dataset}_nodeid_contraint1 ON (n:{dataset}_White) ASSERT n.nodeid IS UNIQUE"
        )
        ret = session.run(  
            f"CREATE CONSTRAINT {dataset}_nodeid_contraint2 ON (n:{dataset}_Black) ASSERT n.nodeid IS UNIQUE"
        )
        # print(ret.single()[0])
        print(ret.single())

        #导入节点
        print(f"Import Nodes ...")
        # CSV文件路径  
        
    
        nlines = buf_count_newlines_gen(nodes_csv_path)-1
        with open(nodes_csv_path, mode='r', encoding='utf-8') as file:  
            print(f"#Nodes: {nlines}")
            reader = csv.DictReader(file)  
            for row in tqdm(reader, total=nlines):  
                node_id = int(row['nodeId'])  
                label = int(row['label'])
                failure = int(row['failure'])
                # feature_0 =row['feature_0']  
                # feature_1 =row['feature_1'] 
                # feature_2 =row['feature_2'] 
                # feature_3 =row['feature_3'] 
                # feature_4 =row['feature_4'] 
                # feature_5 =row['feature_5'] 
                # feature_6 =row['feature_6']  
                # feature_7 =row['feature_7'] 
                # feature_8 =row['feature_8'] 
                # feature_9 =row['feature_9'] 
                # feature_10 =row['feature_10'] 
                # feature_11 =row['feature_11'] 
                # feature_12 =row['feature_12']  
                # feature_13 =row['feature_13'] 
                # feature_14 =row['feature_14'] 
                # feature_15 =row['feature_15'] 
                # feature_16 =row['feature_16'] 
                # feature_17 =row['feature_17'] 
                # feature_18 =row['feature_18']  
                # feature_19 =row['feature_19'] 
                # feature_20 =row['feature_20'] 
                # feature_21 =row['feature_21'] 
                # feature_22 =row['feature_22'] 
                # feature_23 =row['feature_23'] 
                # feature_24 =row['feature_24'] 
                # 使用Cypher查询创建节点  

                cypher = "CREATE (n:User {nodeid: $id, label: $label, feature_0: $feature_0, feature_1: $feature_1, feature_2: $feature_2, feature_3: $feature_3, feature_4: $feature_4, feature_5: $feature_5, feature_6: $feature_6, feature_7: $feature_7, feature_8: $feature_8, feature_9: $feature_9, feature_10: $feature_10, feature_11: $feature_11, feature_12: $feature_12, feature_13: $feature_13, feature_14: $feature_14, feature_15: $feature_15, feature_16: $feature_16, feature_17: $feature_17, feature_18: $feature_18, feature_19: $feature_19, feature_20: $feature_20, feature_21: $feature_21, feature_22: $feature_22, feature_23: $feature_23, feature_24: $feature_24});"

                cypher2 = "CREATE (n:User {nodeid: $id}) RETURN n;"

                if label == 0:
                    cypher3 = f"CREATE (n:{dataset}_White " + "{nodeid: $id, failure: $failure}) RETURN n;"

                elif label == 1:
                    cypher3 = f"CREATE (n:{dataset}_Black " + "{nodeid: $id, failure: $failure}) RETURN n;"

                ret = session.run(  
                        cypher3, 
                        id=node_id,  
                        label=label,
                        failure=failure,
                        # feature_0 =feature_0  ,
                        # feature_1 =feature_1 ,
                        # feature_2 =feature_2 ,
                        # feature_3 =feature_3 ,
                        # feature_4 =feature_4,
                        # feature_5 =feature_5 ,
                        # feature_6 =feature_6 ,
                        # feature_7 =feature_7 ,
                        # feature_8 =feature_8,
                        # feature_9 =feature_9,
                        # feature_10 =feature_10 ,
                        # feature_11 =feature_11 ,
                        # feature_12 =feature_12 ,
                        # feature_13 =feature_13,
                        # feature_14 =feature_14,
                        # feature_15 =feature_15,
                        # feature_16 =feature_16,
                        # feature_17 =feature_17,
                        # feature_18 =feature_18,
                        # feature_19 =feature_19,
                        # feature_20 =feature_20 ,
                        # feature_21 =feature_21,
                        # feature_22 =feature_22,
                        # feature_23 =feature_23,
                        # feature_24 =feature_24  
                )  


        


        # print(f"Creating Index on nodeid")  
        # ret = session.run("DROP INDEX nodeid_index IF EXISTS")
        # ret = session.run(  
        #     "CREATE INDEX nodeid_index FOR (n:User) ON (n.nodeid)"
        # )
        # # print(ret.single()[0])
        # print(ret.single())
  
        #导入边
        print(f"Import Edges ...")
        nlines = buf_count_newlines_gen(edges_csv_path)-1
        with open(edges_csv_path, mode='r', encoding='utf-8') as file:  
            print(f"#Edges: {nlines}")
            reader = csv.DictReader(file)   
            for row in tqdm(reader, total=nlines):  
                srcnodes = int(row['srcnodes'])  
                dstnodes = int(row['dstnodes'])
                edgeId = row['edgeId'] 
                etype = row['etype']  

                # 使用Cypher查询创建关系  #根据自己的关联需求自定义
                cypher = "MATCH (a:User {nodeid: $srcnode}), (b:User {nodeid: $dstnode})  CREATE (a)-[:RELATION  {edgeid: $eid, etype:$et}]->(b)"

                cypher2 = "MATCH (a:User {nodeid:$srcnode}), (b:User {nodeid:$dstnode}) RETURN a,b"
                
                cypher3 = "MERGE (a:User {nodeid:$srcnode}) MERGE (b:User {nodeid:$dstnode}) MERGE (a)-[:RELATION  {edgeid: $eid, etype:$et}]->(b)"

                cypher4 = f"MATCH (a) WHERE (a:{dataset}_White OR a:{dataset}_Black) AND a.nodeid=$srcnode " + f"MATCH (b) WHERE (b:{dataset}_White OR b:{dataset}_Black) AND b.nodeid=$dstnode " + "MERGE (a)-[:RE  {edgeid: $eid, etype:$et}]->(b)"

                ret = session.run(  
                        cypher4,  
                        srcnode=srcnodes,  
                        dstnode=dstnodes, 
                        eid=edgeId,
                        et=etype,
                )  


        ### 修改顶点的failure值
        # print(f"Modify failure value")

        # nlines = buf_count_newlines_gen(nodes_csv_path)-1
        # with open(nodes_csv_path, mode='r', encoding='utf-8') as file:  
        #     print(f"#Nodes: {nlines}")
        #     reader = csv.DictReader(file)  
        #     for row in tqdm(reader, total=nlines):  
        #         node_id = int(row['nodeId'])  
        #         label = int(row['label'])
        #         failure = int(row['failure'])

        #         if label == 0:
        #             cypher3 = "MATCH (n:White) WHERE n.nodeid=$id SET n.failure=$failure RETURN n;"

        #         elif label == 1:
        #             cypher3 = "MATCH (n:Black) WHERE n.nodeid=$id SET n.failure=$failure RETURN n;"

        #         ret = session.run(  
        #             cypher3, 
        #             id=node_id,  
        #             label=label,
        #             failure=failure,
        #         )  

  
    # 现在可以关闭驱动程序了  
    driver.close()