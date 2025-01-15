import pandas as pd
import numpy as np
from typing import Dict

def to_df(node_id_to_community_map: Dict) -> pd.DataFrame: 
    df = pd.DataFrame(columns=['level', 'commid', 'nodeids', 'size'])

    for level in node_id_to_community_map.keys():
        nodeid2commid = node_id_to_community_map[level]

        commids = np.unique(np.array(list(nodeid2commid.values())))
        for commid in commids:
            
            lev = level
            com = commid
            nodeids = []
            size = (np.array(list(nodeid2commid.values())) == commid).sum()

            for nid, cid in nodeid2commid.items():
                if cid == commid:
                    nodeids.append(nid)
            
            df.loc[len(df)] = [lev, com, nodeids, size]
    
    return df


            