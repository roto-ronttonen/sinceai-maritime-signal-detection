#!/bin/bash
set -e
docker compose up datagenerator
docker compose up trainclassifier
docker compose up classifier