# Inventory-management-for-Flipkart
Data preprocessing and exploratory data analysis (EDA) are critical responsibilities for any data science endeavour.
In this post, we'll look at a sample dataset and challenge, as well as apply a few data preprocessing and EDA approaches.

## Introduction
In this project, we analyse a dataset that offers a comprehensive snapshot of an e-commerce inventory. This dataset provides a treasure trove of information, from product attributes and seller details to customer ratings and stock availability.
We analyse the data using Python programming and exploratory data science methods to provide useful insights and answers. Our project aims to provide businesses with the necessary tools to succeed in the competitive e-commerce market, from anticipating stock shortages to revealing the hidden factors influencing consumer preferences.

## The Data
The dataset contains information related to various products available in an e-commerce inventory. Here's a brief overview of the columns:
*	_id: Unique identifier for the product.
*	actual_price: The original price of the product.
*	average_rating: Average rating given by customers.
*	brand: Brand of the product.
*	category: Broad category of the product (e.g., Clothing and Accessories).
*	crawled_at: Date and time the data was crawled.
*	description: Description of the product.
*	discount: Discount offered on the product.
*	images: URLs to the product images.
*	out_of_stock: Boolean flag indicating whether the product is out of stock.
*	pid: Another identifier for the product (perhaps the product ID).
*	product_details: Detailed specifications of the product.
*	seller: Seller of the product.
*	selling_price: The selling price of the product after discount.
*	sub_category: Sub-category of the product (e.g., Bottomwear).
*	title: Title or name of the product.

Data contains total 0f 17 columns , 30000 rows and a total of 510000 data points

| Field Name        | Type     | Non-empty | Empty         |
|-------------------|----------|-----------|---------------|
| Actual Price     | Integer  | 29,137    | 863 (2.88%)   |
| Average Rating    | Decimal  | 27,554    | 2,446 (8.15%) |
| Sub category     | String   | 30,000    | 0 (0%)        |
| Selling Price    | Integer  | 29,998    | 2 (<1%)       |
| Seller           | String   | 28,259    | 1,741 (5.8%)  |
| Product details  | String   | 30,000    | 0 (0%)        |
| Category         | String   | 30,000    | 0 (0%)        |
| Id               | String   | 30,000    | 0 (0%)        |
| Brand            | String   | 27,932    | 2,068 (6.89%) |
| Crawled at       | Datetime | 30,000    | 0 (0%)        |
| Description      | String   | 18,020    | 11,980 (39.93%)|
| Discount         | String   | 29,059    | 941 (3.14%)   |
| Images           | Int      | 30,000    | 0 (0%)        |
| Out of stock     | Boolean  | 30,000    | 0 (0%)        |
| Pid              | String   | 30,000    | 0 (0%)        |
| Title            | String   | 30,000    | 0 (0%)        |
| Url              | Url      | 30,000    | 0 (0%)        |

## Managing Inventory Data Using Python
Now let’s start this task by importing the necessary Python libraries and the dataset we need:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Data1.csv")
df.head()
```
> 
![image](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/ab7d48bf-13ac-4cd1-8082-be454ceebbe1)


## Data Processing Steps
### 1. Conversion of 'crawled_at' to datetime format and extracting date
```
data_processed = data.copy()
data_processed['crawled_at'] = pd.to_datetime(data_processed['crawled_at'], errors='coerce')
data_processed['crawled_date'] = data_processed['crawled_at'].dt.date

```
### 2. Extraction of discount percentages
```
data_processed['discount_percentage'] = data_processed['discount'].str.extract('(\d+)').astype(float)
```
