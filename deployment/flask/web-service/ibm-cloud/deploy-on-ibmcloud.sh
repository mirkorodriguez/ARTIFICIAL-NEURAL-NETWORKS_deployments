#!/usr/bin/env bash

# Copy service script
cp -f ../service.py service.py

# Copy model loader script
cp -f ../ann_loader.py ann_loader.py

# Copy model .h5 to models/
SOURCE="../../../../models/classification/churn-clients"
DESTINATION="models/classification/churn-clients"
mkdir -p $DESTINATION
ls -l $SOURCE
cp -rf "$SOURCE/"* $DESTINATION

# Replace MODEL_PATH on ann_loader.py
# LINE_TO_REPLACE=$(grep -n 'MODEL_PATH =' ann_loader.py | gawk '{print $1}' FS=":")
LINE_TEXT_REPLACE="    MODEL_PATH = \"models\/classification\/churn-clients\""
sed -i "$(grep -n 'MODEL_PATH =' ann_loader.py | gawk '{print $1}' FS=":")s/.*/$LINE_TEXT_REPLACE/" ann_loader.py

# Cloud Foundry login
cf login -a https://api.ng.bluemix.net

# Cloud Foundry deploy
cf push &&

# Deleting tmp files
rm -rf service.py
rm -rf ann_loader.py
rm -rf models
