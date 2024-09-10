#!/bin/bash

chmod +x eval.sh && ./eval.sh
chmod +x offline_data_split.sh && ./offline_data_split.sh

cd ../neo4j

chmod +x run.sh && ./run.sh

cd -