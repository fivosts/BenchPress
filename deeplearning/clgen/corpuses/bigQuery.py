"""BigQuery API to fetch github data."""
import os
import pathlib

from google.cloud import bigquery
from eupy.native import logger as l

# Construct a BigQuery client object.
client = bigquery.Client()

count_query = """
SELECT COUNT(*)
FROM `bigquery-public-data.github_repos.files`
WHERE substr(path, -3, 4) = '.cl'
"""

db_query = """
SELECT file.repo_name, file.path, file.ref, file.mode, 
       file.id, file.symlink_target, contentfile.size, 
       contentfile.content, contentfile.binary, contentfile.copies
FROM `bigquery-public-data.github_repos.contents` as contentfile
INNER JOIN `bigquery-public-data.github_repos.files` as file ON file.id = contentfile.id AND substr(file.path, -3, 4) = '.cl'
LIMIT 10
"""

count_job = client.query(count_query)  # Make an API request.
file_job  = client.query(db_query)  # Make an API request.

for row in count_job:
  for item in row:
    l.getLogger().info(item)
for row in file_job:
  for item in row:
    l.getLogger().info(item)
