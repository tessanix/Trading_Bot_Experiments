#!/bin/bash

# Define the source directory where your Python project resides
SOURCE_DIR="/home/tessan/dev/Trading_bot_and_RL"

# Define the destination directory where you want to place the compressed package
DEST_DIR="/home/tessan/dev/Trading_bot_and_RL"

# Define the name of the compressed package
PACKAGE_NAME="production_package.tar.gz"

# Create a temporary directory to store the production-ready files

mkdir $SOURCE_DIR/tmp

TEMP_DIR=$SOURCE_DIR/tmp/bot_server
# Copy production-ready files and folders to the temporary directory
# Adjust the paths as per your project structure

rsync -av $SOURCE_DIR/server $TEMP_DIR
rsync -av $SOURCE_DIR/utils $TEMP_DIR
rsync -av $SOURCE_DIR/xtb_api $TEMP_DIR
rsync -av --exclude 'Moving_Average_Strategy.py' --exclude 'strategy_tester_loop.py' $SOURCE_DIR/strategies $TEMP_DIR
rsync -av --exclude 'tests' --exclude 'ac_training_loop.py' --exclude 'ac_trained_loop.py' --exclude 'ac_training_tester.ipynb' --exclude 'ac_trained_tester.ipynb' --exclude 'model_1_613_episodes' $SOURCE_DIR/reinforcement_learning $TEMP_DIR

#removes __pycache__ dir
find $TEMP_DIR -type d -name '__pycache__' -exec rm -rf {} +

# rsync -av --exclude 'Movin' --exclude 'folder_to_exclude2' /path/to/source_directory/ /path/to/destination_directory/

# Navigate to the temporary directory
cd $TEMP_DIR || exit

# Create the compressed package
tar -czvf "$DEST_DIR/$PACKAGE_NAME" .

# Clean up the temporary directory
# rm -rf $TEMP_DIR

echo "Production package created: $DEST_DIR/$PACKAGE_NAME"