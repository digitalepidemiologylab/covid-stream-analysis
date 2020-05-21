#!/bin/sh


aws s3 sync s3://crowdbreaks-covid-stream/stream/ data/raw --profile private --region eu-central-1
