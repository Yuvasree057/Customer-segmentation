# Customer Segmentation Report

## Overview
We performed customer segmentation using KMeans clustering on the Mall Customers dataset. The features used for clustering were `Annual Income (k$)` and `Spending Score (1-100)`.

## Methodology
- Standardized features to ensure equal weighting.
- Used the Elbow Method to determine the optimal number of clusters, which was identified as **5**.

## Cluster Characteristics
               Age  Annual Income (k$)  Spending Score (1-100)  Count
Cluster                                                              
0        42.716049           55.296296               49.518519     81
1        32.692308           86.538462               82.128205     39
2        25.272727           25.727273               79.363636     22
3        41.114286           88.200000               17.114286     35
4        45.217391           26.304348               20.913043     23

## Analysis of Purchase Patterns & Demographics
- **Cluster 0**: Average Income, Average Spending. The largest segment (81 customers), representing typical middle-class, middle-age mall-goers.
- **Cluster 1**: High Income, High Spending. These are young adults (average age 32) who have a high spending score. They are the ideal target demographic for premium, luxury products and high-value marketing campaigns.
- **Cluster 2**: Low Income, High Spending. Very young demographic (average age 25) who spend a lot despite lower income. These could be students or young professionals who spend freely. 
- **Cluster 3**: High Income, Low Spending. Older demographic (average age 41) who are careful with their money despite having high annual incomes. They might require targeted discounts or high-value propositions to increase their spending.
- **Cluster 4**: Low Income, Low Spending. Older demographics who are highly price-sensitive. 

## Key Insights
1. The most lucrative segment is **Cluster 1**, as they have the disposable income and the willingness to spend.
2. **Cluster 3** represents an untapped opportunity; they have the money but aren't spending it. Understanding their preferences could unlock significant revenue.
3. **Cluster 2** requires careful engagement as they spend highly relative to their income. 
