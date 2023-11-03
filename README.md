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
### 1. Column Datatypes
```
df.dtypes
```
>_id                 object
>
>actual_price        object
>
>average_rating     float64
>
>brand               object
>
>category            object
>
>crawled_at          object
>
>description         object
>
>discount            object
>
>images              object
>
>out_of_stock          bool
>
>pid                 object
>
>product_details     object
>
>seller              object
>
>selling_price       object
>
>sub_category        object
>
>title               object
>
>url                 object
>
>dtype: object


### 2.Dropping Unwanted Columns
```
df.drop("_id", axis=1, inplace=True)
df.drop("description", axis=1, inplace=True)
df.drop("url", axis=1, inplace=True)
```
### 3. Checking Null Values
```
df.isna().sum()
```
>actual_price        863
>
>average_rating     2446
>
>brand              2068
>
>category              0
>
>crawled_at            0
>
>discount            941
>
>images                0
>
>out_of_stock          0
>
>pid                   0
>
>product_details       0
>
>seller             1741
>
>selling_price         2
>
>sub_category          0
>
>title                 0
>
>dtype: int64

### 4. Converting Datatypes of actual_price, average_rating, crawled_at and discount
```
df['actual_price'] = df['actual_price'].str.replace(',', '').astype(float)
df['selling_price'] = df['selling_price'].str.replace(',', '').astype(float)
df['crawled_at'] = pd.to_datetime(df['crawled_at'], errors='coerce')
df['discount'] = df['discount'].str.extract('(\d+)').astype(float)
```
```
df.dtypes
```
>actual_price              float64
>
>average_rating            float64
>
>brand                      object
>
>category                   object
>
>crawled_at         datetime64[ns]
>
>discount                  float64
>
>images                     object
>
>out_of_stock                 bool
>
>pid                        object
>
>product_details            object
>
>seller                     object
>
>selling_price             float64
>
>sub_category               object
>
>title                      object
>
>dtype: object


### 5. Advance Data Preproccesing Techniques
### Imputing Missing values in "brand" with most Frequent brands 

```
df['brand'] = df['brand'].str.capitalize()

most_frequent_brand = df['brand'].mode()[0]

df['brand'].fillna(most_frequent_brand, inplace=True)

mode_seller = df['seller'].mode()[0]
df['seller'].fillna(mode_seller, inplace=True)
```
### Using KNN Imputer Imputing Missing values in 'actual_price', 'average_rating', 'discount' and 'selling_price'
```
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Select the columns with missing values
columns_with_missing = ['actual_price', 'average_rating', 'discount', 'selling_price']

# Create a new DataFrame with only the selected columns
data_to_impute = df[columns_with_missing]

# Initialize the KNNImputer
imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors (n_neighbors) as needed

# Impute missing values
data_imputed = imputer.fit_transform(data_to_impute)

# Create a DataFrame with imputed values
data_imputed_df = pd.DataFrame(data_imputed, columns=columns_with_missing)

# Replace the columns in the original data with the imputed values
df[columns_with_missing] = data_imputed_df
```

```
df.isnull().sum()
```
>actual_price       0
>
>average_rating     0
>
>brand              0
>
>category           0
>
>crawled_at         0
>
>discount           0
>
>images             0
>
>out_of_stock       0
>
>pid                0
>
>product_details    0
>
>seller             0
>
>selling_price      0
>
>sub_category       0
>
>title              0
>
>dtype: int64

### Creating New Features from Existing Features

### Creating Crawled_date column
```
df['crawled_date'] = df['crawled_at'].dt.date
```
###  Extract 'Color' and 'Size' attributes
```
import ast
# Convert the string representation of lists to actual lists
df['product_details_list'] = df['product_details'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

df['color'] = df['product_details_list'].apply(lambda x: next((item['Color'] for item in x if 'Color' in item), None))
df['size'] = df['product_details_list'].apply(lambda x: next((item['Size'] for item in x if 'Size' in item), None))
```
```
# 5. Tokenization of product titles
from collections import Counter
import re

# Function to preprocess and tokenize text
def tokenize(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    return tokens

# Tokenize product titles
tokens_list = df['title'].apply(tokenize)

# Flatten list of lists and count word occurrences
word_counts = Counter(word for tokens in tokens_list for word in tokens)

# Get the most common words in product titles
common_keywords = word_counts.most_common(20)
common_keywords
```
>[('men', 28001),
 ('tshirt', 13416),
 ('neck', 12894),
 ('solid', 10150),
 ('round', 8912),
 ('printed', 7193),
 ('pack', 4925),
 ('of', 4920),
 ('blue', 4796),
 ('fit', 4043),
 ('black', 3893),
 ('shirt', 3688),
 ('polo', 3145),
 ('slim', 3039),
 ('casual', 3011),
 ('multicolor', 2451),
 ('white', 2314),
 ('sleeve', 2264),
 ('full', 2249),
 ('collar', 2150)]


## 2. Exploratory DATA analysis
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Pastel1', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

```
![image](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/8189d597-0163-4c8b-a550-9898caff2882)

```
df["out_of_stock"].value_counts()
```
>False    28258
True      1742
Name: out_of_stock, dtype: int64

### Top 10 brands by out of stock
```
out_of_stock_percentage = df.groupby('brand')['out_of_stock'].mean().sort_values(ascending=False)
```
```
import plotly.express as px

# Create a bar plot using Plotly for the top out-of-stock brands
top_out_of_stock_brands = out_of_stock_percentage.reset_index().head(10)
fig = px.bar(top_out_of_stock_brands, x='out_of_stock', y='brand', orientation='h', color='brand')
fig.update_layout(
    title='Top 10 Brands by Out-of-Stock Percentage',
    xaxis_title='Out-of-Stock Percentage',
    yaxis_title='Brand'
)

# Show the plot
fig.show()

```
![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/7751c5c2-e0f4-44d6-875b-c0c75bd41032)
### Plot between Category and Out of Stock
```
import pandas as pd
import plotly.express as px

# Group by 'category' and count the number of out-of-stock and in-stock products
category_out_of_stock_counts = df.groupby(['category', 'out_of_stock']).size().reset_index(name='count')

# Create a bar chart using Plotly
fig = px.histogram(
    category_out_of_stock_counts,
    x='category',
    y='count',
    color='out_of_stock',
    barmode='group',
    title='Distribution of Out-of-Stock Products by Category',
    labels={'category': 'Category', 'count': 'Count', 'out_of_stock': 'Out of Stock'},
)

# Set layout options (e.g., axis labels, title, etc.)
fig.update_layout(xaxis_title='Category', yaxis_title='Count')

# Show the plot
fig.show()

```
![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/678a7597-d6b5-454b-835d-525b4515d784)

Inferences : Brands with a high percentage of out-of-stock products may have issues with inventory management.

### Predicting Out of Stock 

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Select features and target variable
X = df[['title','brand', 'size', 'color']]
y = df['out_of_stock']

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['title','brand','size','color',])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a classification model (Random Forest Classifier)
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['In Stock', 'Out of Stock'])

print(f"Accuracy: {accuracy}")
print(report)

```
>Accuracy: 0.955
>
|           | Precision |  Recall  | F1-Score | Support |
|-----------|:---------:|:--------:|:--------:|:-------:|
| In Stock  |   0.96    |   0.99   |   0.98   |   5642  |
| Out of Stock |   0.71    |   0.41   |   0.52   |   358   |
| Accuracy  |           |          |   0.95   |   6000  |
| Macro Avg |   0.84    |   0.70   |   0.75   |   6000  |
| Weighted Avg | 0.95    |   0.95   |   0.95   |   6000  |

### Price Prediction 
```
import plotly.express as px

# Create a histogram using Plotly
fig = px.histogram(df, x='average_rating', nbins=30, title='Distribution of Product Ratings', opacity=0.7, color_discrete_sequence=['skyblue'])

# Customize the appearance of the histogram (optional)
fig.update_traces(marker_line_color='black')

# Set layout options (e.g., title, x-axis label, y-axis label)
fig.update_layout(xaxis_title='Average Rating', yaxis_title='Number of Products')

# Show the plot
fig.show()

```
![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/0e837bc5-b518-47b9-a9b8-1362aefcaee1)

```
import plotly.express as px

# Create a histogram plot using Plotly for 'actual_price_num'
fig = px.histogram(df, x='actual_price', nbins=50, color_discrete_sequence=['lightcoral'])
fig.update_layout(
    title='Distribution of Product Actual Price',
    xaxis_title='Actual Price',
    yaxis_title='Number of Products'
)

# Show the plot
fig.show()

```
![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/1b8396c8-b328-4cf2-a4a9-bc9f0cd7dd44)

```
import plotly.express as px

# Create a histogram plot using Plotly for 'discount_percentage'
fig = px.histogram(df, x='discount', nbins=30, color_discrete_sequence=['goldenrod'])
fig.update_layout(
    title='Distribution of Discount Percentage',
    xaxis_title='Discount Percentage',
    yaxis_title='Number of Products'
)

# Show the plot
fig.show()

```
![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/cedfcfc2-20ea-4c08-9fce-bf7458b4bd47)

```
import plotly.express as px


# Create a scatter plot using Plotly
fig = px.scatter(df, x='selling_price', y='average_rating', title='Scatter Plot of Selling Price vs. Rating')

# Customize the appearance of the scatter plot (optional)
fig.update_traces(marker=dict(size=5, opacity=0.5), selector=dict(mode='markers+text'))

# Set layout options (e.g., axis labels, title, etc.)
fig.update_layout(xaxis_title='Selling Price', yaxis_title='Average Rating')

# Show the plot
fig.show()

```
![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/2271aa02-6c5c-4d40-ae6c-88f2981b3e83)

```
import plotly.express as px

# Assuming you have a DataFrame called 'df' with columns 'color' and 'selling_price'
# Replace 'df' with your actual DataFrame name

# Create a histogram using Plotly for 'color' and 'selling_price'
fig = px.histogram(df, x='color', y='selling_price', title='Histogram of Selling Price by Color',
                   color_discrete_sequence=['skyblue'])

# Set layout options (e.g., title, axis labels)
fig.update_layout(xaxis_title='Color', yaxis_title='Selling Price')

# Show the plot
fig.show()

```

![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/bfbc9018-504e-4ab0-a276-1637843734b0)

```
import plotly.express as px

# Assuming you have a DataFrame called 'df' with columns 'size' and 'selling_price'
# Replace 'df' with your actual DataFrame name

# Create a histogram using Plotly for 'size' and 'selling_price'
fig = px.histogram(df, x='size', y='selling_price', title='Histogram of Selling Price by Size',
                   color_discrete_sequence=['lightcoral'])

# Set layout options (e.g., title, axis labels)
fig.update_layout(xaxis_title='Size', yaxis_title='Selling Price')

# Show the plot
fig.show()

```

![newplot](https://github.com/khushinvyas/Inventory-management-for-Flipkart/assets/120413040/09a48d19-769c-45c2-8d5e-1b096bb4e3a0)

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Define the features (independent variables) and target (price)
features = ['actual_price', 'average_rating', 'discount', 'size', 'color','seller']
target =  'selling_price'

# Select the features and target
X = df[features]
y = df[target]

# Convert 'size' and 'color' columns to numeric using label encoding
label_encoder = LabelEncoder()
X['size'] = label_encoder.fit_transform(df['size'])
X['color'] = label_encoder.fit_transform(df['color'])
X['seller'] = label_encoder.fit_transform(df['seller'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Regressor model (you can try other regression models too)
model2 = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model2.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model2.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Optionally, you can use the trained model for price predictions
new_data = pd.DataFrame({
    'actual_price': [25000],
    'average_rating': [4.5],
    'discount': [20],
    'selling_price': [22000],
    'size': ['Medium'],  # Replace with the actual size value
    'color': ['Blue']   # Replace with the actual color value
})
```

>Mean Squared Error: 473.8517457611935
>
>Root Mean Squared Error: 21.768136019448093
