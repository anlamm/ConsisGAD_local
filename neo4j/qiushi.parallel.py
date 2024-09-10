import csv  
from neo4j import GraphDatabase  
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time


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


def insert_edge_kernel(srcnode, dstnode):
    cypher3 = "MERGE (a:User {nodeid:$srcnode}) MERGE (b:User {nodeid:$dstnode}) MERGE (a)-[:RELATION]->(b)"
    # cypher3 = "CREATE (n:User {nodeid: $id}) RETURN n;"
    # print(f"insert {srcnode},{dstnode}")
    ret=session.run(  
        cypher3,  
        srcnode=int(srcnode),  
        dstnode=int(dstnode),
        id=int(srcnode),
    )  
    # print(ret.single())

    # print(f"insert {srcnode},{dstnode}")

    return 1



def insert_node_kernel(nodeid):
    cypher2 = "CREATE (n:User {id: $id, nodeid: $id}) RETURN n;"
    print(f"inserting {nodeid}")

    # session = driver.session(database=database) 
    session.run(  
        cypher2,  
        id=int(nodeid),
    )  
    # session.close()

    return 1


def batches(edgelines):
    pass


def log_result(result):
    pass

if __name__ == "__main__":
  
    # 连接到Neo4j数据库  
    uri = "bolt://localhost:7687"  
    user = "neo4j"    #neo4j用户名
    password = "asdf123456789"   #neo4j密码
    driver = GraphDatabase.driver(uri, auth=(user, password))  
    database = 'neo4j'
    nodes_csv_path = '/home/yliumh/github/ConsisGAD/data/CSV/amazon_node.csv' 
    edges_csv_path = '/home/yliumh/github/ConsisGAD/data/CSV/amazon_edge.csv' 

    session = driver.session(database=database)

    session.run(  
        "MATCH (n) DETACH DELETE n;"
    )

    print(f"Creating constrain on nodeid")
    ret = session.run("DROP INDEX nodeid_index IF EXISTS") 
    ret = session.run("DROP CONSTRAINT nodeid_contraint IF EXISTS")
    ret = session.run(  
        "CREATE CONSTRAINT nodeid_contraint ON (n:User) ASSERT n.nodeid IS UNIQUE"
    )
    print(ret.single())

    #导入节点
    print(f"Import Nodes ...")
    # CSV文件路径  
        
    
    nlines = buf_count_newlines_gen(nodes_csv_path)-1
    with open(nodes_csv_path, mode='r', encoding='utf-8') as file:  
        print(f"#Nodes: {nlines}")
        lines = file.readlines()[1:] 
        lines = [[line.split(',')[0]] for line in lines]
        print(lines[0], len(lines))

        p = Pool()
        time_st = time.time()
        for line in lines:
            res = p.apply_async(insert_node_kernel, args=(line), callback=log_result)
            # _ = res.get()
            # partial(insert_node_kernel, session)(line[0])
        p.close()
        p.join()
        time_ed = time.time()
        print(f"Total time: {time_ed-time_st:.2f} sec")

     
    ret = session.run(  
        "MATCH (n) return count(n)"
    )
    print(f"#Inserted Nodes: {ret.single()}")

    print(f"Creating constrain on nodeid")
    ret = session.run("DROP INDEX nodeid_index IF EXISTS") 
    ret = session.run("DROP CONSTRAINT nodeid_contraint IF EXISTS")
    ret = session.run(  
        "CREATE CONSTRAINT nodeid_contraint ON (n:User) ASSERT n.nodeid IS UNIQUE"
    )
    print(ret.single())


    #导入边
    print(f"Import Edges ...")
    nlines = buf_count_newlines_gen(edges_csv_path)-1
    with open(edges_csv_path, mode='r', encoding='utf-8') as file:  
        print(f"#Edges: {nlines}")
        lines = file.readlines()[1:] ## 
        lines = [line.split(',')[1:3] for line in lines]
        print(lines[0], len(lines))

        p = Pool()
        time_st = time.time()
        for line in lines:
            res = p.apply_async(insert_edge_kernel, args=(line))
            _ = res.get()
            # partial(insert_node_kernel, session)(line[0])
        p.close()
        p.join()
        time_ed = time.time()
        print(f"Total time: {time_ed-time_st:.2f} sec")
    
 
    ret = session.run(  
        "MATCH ()-->() RETURN count(*)"
    )
    print(f"#Inserted Edges: {ret.single()}")
  
    # 现在可以关闭驱动程序了  
    session.close()
    driver.close()