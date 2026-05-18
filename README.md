# Customer Segmentation Using RFM Analysis and Unsupervised Machine Learning

**Author:** Muhammad Fiaz  
**Domain:** Consumer Segmentation and Consumer Behavior  
**Platform:** Google Colab  
**Language:** Python 3  
**Dataset:** UCI Online Retail Dataset via Kaggle  

---

## Overview

This project applies the RFM (Recency, Frequency, Monetary) behavioral framework combined
with three unsupervised machine learning algorithms to segment customers of a UK-based
e-commerce retailer. The goal is to identify distinct customer groups from raw transactional
data and derive actionable marketing strategies for each segment.

The analysis covers 392,692 cleaned transactions from December 2010 to December 2011,
spanning 4,338 unique customers across 37 countries.

---

## Research Questions

- How can raw transactional data be transformed into behavioral customer profiles using RFM?
- Which clustering algorithm produces the most statistically valid and interpretable segments?
- What are the managerial implications of data-driven segmentation for targeted marketing?

---

## Dataset

- Source: UCI Machine Learning Repository via Kaggle
- Link: https://www.kaggle.com/datasets/carrie1/ecommerce-data
- Original size: 541,909 rows, 8 columns
- Final size after cleaning: 392,692 rows
- Fields: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## Project Structure

```
.
├── Customer_Segmentation_RFM.ipynb    # Main Google Colab notebook
├── customer_segments.csv              # Final segmented customer output
└── README.md                          # Project documentation
```

---

## Notebook Structure

1. Environment Setup and Library Imports
2. Data Loading via Kaggle API
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
5. RFM Feature Engineering
6. Log Transformation and Scaling
7. K-Means Clustering
8. Hierarchical Clustering
9. DBSCAN Clustering
10. Algorithm Comparison and Evaluation
11. Business Interpretation and Managerial Insights
12. Conclusion

---

## Methods

**Preprocessing**
- Removed 135,080 rows with missing CustomerID
- Removed 5,225 duplicate rows
- Removed 8,872 cancelled transactions
- Removed 40 rows with invalid Quantity or UnitPrice
- Converted InvoiceDate to datetime and CustomerID to integer

**Feature Engineering**
- Computed Recency, Frequency, and Monetary values per customer
- Applied log1p transformation to correct positive skewness
- Applied RobustScaler for outlier-resistant standardization

**Clustering Algorithms**
- K-Means with k-means++ initialization, k=4
- Agglomerative Hierarchical Clustering with Ward linkage, k=4
- DBSCAN with eps=0.25 and min_samples=5

**Evaluation Metrics**
- Silhouette Score
- Davies-Bouldin Score
- Calinski-Harabasz Score

---

## Results

### Algorithm Comparison

| Algorithm       | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|-----------------|------------------|----------------|-------------------|
| K-Means (k=4)   | 0.3340           | 1.0156         | 3286.68           |
| Hierarchical    | 0.2275           | 1.2293         | 2534.61           |
| DBSCAN          | 0.1034           | 1.2507         | 1037.86           |

K-Means was selected as the final model based on best performance across all three metrics.

### Customer Segments

| Segment         | Customers    | Avg Recency | Avg Frequency | Avg Monetary | Revenue Share |
|-----------------|--------------|-------------|---------------|--------------|---------------|
| Champions       | 727 (16.76%) | 13 days     | 14 orders     | GBP 8,042    | 65.79%        |
| Loyal Customers | 1,220 (28.12%)| 76 days    | 4 orders      | GBP 1,714    | 23.52%        |
| New Customers   | 839 (19.34%) | 18 days     | 2 orders      | GBP 560      | 5.29%         |
| Lost Customers  | 1,552 (35.78%)| 183 days   | 1 order       | GBP 309      | 5.40%         |

Champions represent 16.76% of customers and generate 65.79% of total revenue.

---

## Key Findings

- Total revenue across the period was GBP 8,887,208.89
- Revenue peaks in November 2011, consistent with holiday retail patterns
- The United Kingdom accounts for approximately 82% of total revenue
- All three RFM metrics exhibit heavy right skew requiring log transformation
- K-Means outperforms Hierarchical Clustering and DBSCAN on all evaluation metrics
- A strong Pareto effect is present: a small minority of customers drive the majority of revenue

---

## How to Run

1. Open the notebook in Google Colab
2. Run the first code cell to install dependencies
3. Upload your kaggle.json API token when prompted
4. Run all remaining cells sequentially

To obtain your Kaggle API token:
- Go to https://www.kaggle.com/settings
- Scroll to the API section
- Click Create New Token
- Upload the downloaded kaggle.json file when prompted in the notebook

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
yellowbrick
```

All dependencies except yellowbrick are pre-installed in Google Colab.
yellowbrick is installed automatically in the first cell.

---

## Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import KElbowVisualizer
```

---

## Output

The notebook produces a segmented customer file `customer_segments.csv` with the following columns:

| Column     | Description                              |
|------------|------------------------------------------|
| CustomerID | Unique customer identifier               |
| Recency    | Days since last purchase                 |
| Frequency  | Total number of unique orders            |
| Monetary   | Total revenue generated (GBP)            |
| Segment    | Assigned segment label                   |
