# lab3
Project Overview
This project demonstrates the implementation of a Medallion Architecture (Lakehouse) data pipeline using Azure Databricks and PySpark. The goal of the pipeline is to process raw Amazon Electronics review data and product metadata, transforming it from raw JSON into a curated, ML-ready dataset.

Architecture & Technologies
The pipeline follows the industry-standard Medallion Architecture to ensure data quality and reliability:

Bronze Layer (Raw): Stores the original data in its native format (JSON) in Azure Data Lake Storage (ADLS) Gen2.

Silver Layer (Processed): Contains cleaned, filtered, and structured data stored in Parquet format. This layer removes invalid records and standardizes schemas.

Gold Layer (Curated): The final "Enriched" dataset where reviews are joined with product metadata (price, brand, title) to create a feature set ready for analytics and machine learning.

Core Technologies:
Azure Databricks: A unified analytics platform for data engineering and machine learning.

Apache Spark (PySpark): The distributed processing engine used for large-scale data transformation.

Parquet: A columnar storage file format that provides efficient data compression and faster query performance compared to JSON.

Azure Data Lake Storage (ADLS) Gen2: The scalable storage layer used to host the different data containers (raw, processed, curated).

Pipeline Workflow
The project is divided into three modular notebooks orchestrated via Databricks Jobs:

1. Data Loading & Initial Cleaning (01_load_reviews)
Connects to ADLS Gen2 using storage account keys.

Loads raw review data from the Silver layer (Parquet).

Filters out rows with missing IDs or ratings.

Enforces business rules (e.g., ratings must be between 1-5, review text must be > 10 characters).

Saves the result back to the processed container.

2. Data Enrichment (02_enrich_with_metadata)
Loads the raw product metadata (JSON) from the raw container.

Performs a Left Join between the cleaned reviews and product metadata on the asin (product ID) column.

Selects specific features such as brand, price, and title.

3. Curated Feature Engineering (03_write_gold_features_v1)
Selects the final list of columns required for downstream ML models.

Writes the final dataset to the curated container in the Gold layer.

Automation: Databricks Jobs
The notebooks are orchestrated using Databricks Jobs, which manages dependencies between tasks.

Task 1 must succeed before Task 2 begins.

This ensures that the enrichment step always has the most recent cleaned data available.

How to Run
Configure Storage: Ensure your storage account name and access keys are updated in the configuration cell of each notebook.

Cluster: Attach the notebooks to a Databricks cluster (Runtime 10.4 LTS or higher recommended).

Job Execution: Create a new Workflow in Databricks, link the notebooks in sequence, and click "Run Now."

Future Enhancements
Sentiment Analysis: Use Spark NLP to generate sentiment scores for the reviewText in the Silver layer.

Delta Lake: Implement Delta tables instead of standard Parquet to allow for ACID transactions and time travel (versioning).

Schema Evolution: Implement checks to handle changes in the raw JSON metadata structure over time.
