# Size-Constrained Clustering


## Hierarchical Leiden

reference: [graspologic.partition.hierarchical_leiden](https://github.com/graspologic-org/graspologic/blob/0aab4e0dd9a3387b0d761b4a0bb7ffd2a1a5ea25/graspologic/partition/leiden.py#L417)

与函数 :func:graspologic.partition.leiden 不同的是，该函数不会在实现最大化后停止。在某些大型图中，找出成员数超过 max_cluster_size 的特别大的群落并从该群落中单独生成一个子网络是非常有用的。然后将这个子网络视为一个完全独立的实体，在其上运行 leiden，然后将新的、更小的群落映射到原始群落图空间中。结果也有很大不同；返回的 List[HierarchicalCluster] 更像是每个层级的状态对数。第 0 层的所有 HierarchicalClusters 都应视为运行 :func:graspologic.partition.leiden 的结果。然后，每个成员数量大于 max_cluster_size 的社区也会有 level == 1 的条目，依此类推，直到没有社区的成员数量大于 max_cluster_size，或者我们无法再进一步细分。