#!/bin/bash
# send_node.sh
NODES=("MPI-node9" "MPI-node5" "MPI-node15" "MPI-node14" "MPI-node3")

SOURCE_DIR=~/CO3067_HK251_Group_03

DEST_DIR=/root/

for node in "${NODES[@]}"; do
    echo "Copying to $node..."
    scp -r "$SOURCE_DIR" root@"$node":"$DEST_DIR"
done

echo "Done."