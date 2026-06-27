#!/bin/bash
# Chay pipeline End-to-End theo guideline BTC: doc /code/private_test.json -> /code/submission*.csv
set -e
cd /app
python predict.py
