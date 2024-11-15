#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import re
import plotly.graph_objs as go


# In[2]:


file=os.getcwd()
file_name = 'Amit_Rapid_Test_ Kits'
absolute_path = os.path.abspath(file_name)
print(absolute_path)
full_path=os.path.join(absolute_path+'.xlsx')
dirname=os.path.dirname(os.path.abspath('Amit Rapid Test Kits'))
dirname


# In[3]:


df=pd.read_excel(full_path)
df.tail(5)


# In[4]:


# pd.set_option('display.float_format', '{:.2f}'.format)
df.describe()


# In[5]:


df1=df.copy()


# In[6]:


pattern=r"\bNon-Vaccum\b|\b'Vaccum'\b'"
pattern = r'\bNon-Vacuum\b|\bVacuum\b'

# Apply lambda function to extract the matched pattern
df1['variant'] = df1['products'].apply(lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else None)


# Print the result
df1


# In[7]:


df1['products'][1]


# In[8]:


pattern1 = r'(?<=Tubes\s)(\w+\s\w+)|(?<=Tubes\s)(\s\d\.\d+\s\S\s\w+\s\w+)'

df1['products_variant'] = df1['products'].apply(lambda x: re.search(pattern1, x).group() if re.search(pattern1, x) else '')
df1['products_variant']
df1.head(5)


# In[9]:


# Updated regex pattern
pattern1 = r'(?<=Tubes\s)(\w+\s\w+)|Tubes\s(\d\.\d+\s\S+\s\w+\s\w+)'

# Apply the regex pattern to extract the desired part
df1['products_variant'] = df1['products'].apply(
    lambda x: re.search(pattern1, x).group(1) if re.search(pattern1, x) and re.search(pattern1, x).group(1) else 
              (re.search(pattern1, x).group(2) if re.search(pattern1, x) and re.search(pattern1, x).group(2) else '')
)

# Display the DataFrame
df1.head(5)


# In[10]:


pattern2 = r'(\d)\s+milliliter'       # Captures a single digit before "milliliter"
pattern3 = r'(\d\.\d)\s+milliliter'   # Captures a decimal number before "milliliter"

# Apply the patterns to the 'products' column and create a new 'size' column
df1['size'] = df1['products'].apply(
    lambda x: (match := re.search(pattern3, x)) and match.group(1) or (
        (match := re.search(pattern2, x)) and match.group(1) or ''
    )
)

df1


# In[11]:


product_variant=df1.products_variant.value_counts()


# In[12]:


product_variant= px.bar(product_variant, x=product_variant.index, y=product_variant.values)
product_variant


# In[13]:


df1.contract_date.value_counts()


# In[14]:


df1['variant'].value_counts()


# In[ ]:





# In[15]:


date_format = "%d/%m/%Y %H:%M"
df1['Date'] = pd.to_datetime(df1['contract_date'], format=date_format)
df1.Date


# In[16]:


# Extract day, month, and year into separate columns
df1['Date1'] = df1['Date'].dt.date
df1['Month'] = df1['Date'].dt.month
df1['Year'] = df1['Date'].dt.year
df1['Week'] = df1['Date'].dt.week


# In[17]:


df1['Date1']


# In[18]:


#Tenders per day
tenders_per_day=df1.groupby(df1['Date1']).size()
tenders_per_day



# In[19]:


date=pd.DataFrame({'Date1':tenders_per_day.index, 'Count':tenders_per_day.values})
date


# In[20]:


date_plot = px.line(date, x='Date1', y='Count', title='Number of Tenders Per Day', labels={'count': 'Number of Tenders', 'date': 'Date'})
date_plot


# In[21]:


# Tenders per Month
df1.columns
tenders_per_month=df1.groupby(df1['Month']).size()
tenders_per_month



# In[22]:


Month=pd.DataFrame({'Month':tenders_per_month.index,'Count':tenders_per_month.values})
Month


# In[23]:


month_plot = px.line(Month, x='Month', y='Count', title='Number of Tenders Per Month', labels={'count': 'Number of Tenders', 'date': 'Date'})
month_plot


# In[24]:


#Tenders Per Weeek 
tenders_per_week=df1.groupby(df1['Week']).size()
Week=pd.DataFrame({'Week':tenders_per_week.index,'Count':tenders_per_week.values})
Week


# In[25]:


week_plot = px.line(Week, x='Week', y='Count', title='Number of Tenders Per Week', labels={'count': 'Number of Tenders', 'date': 'Date'})
week_plot


# In[26]:


df2_pred=df1.copy()


# Verify the columns have been dropped
df2_pred.head()


# In[27]:


null_count=df2_pred.isnull().sum()


# In[28]:


# Fill NA
for  v,i in null_count.items():
    percentage = round(i / 614 * 100)
    print(v, percentage)



# In[29]:


df2_pred
    
    
    
    
    


# In[30]:


df1['MinUnitPrice']=df1['Total']/df1['Quantities']
df2_pred['MinUnitPrice']=df2_pred['Total']/df2_pred['Quantities']


# In[31]:


MUP_group=df2_pred.groupby(['variant','products_variant','size'])['MinUnitPrice'].median()
MUP_group


# In[32]:


MUP_group.unstack().plot(kind='bar', ylim=(0, 20), width=1.0, figsize=(10, 6))
# Set labels and title
plt.ylabel('Median Unit Price')
plt.title('Median Unit Price by Variant and Product Variant')

# Place the legend outside the plot
plt.legend(title='Products Variant', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust the plot to make room for the legend
plt.show()


# In[33]:


df3=df2_pred.copy()
df3.drop(['Ministry','contract_number','office_zone','Buyer_Designation','Prices','Date','Year','contract_date','Organization_name','Department','Date1','products','Month','Week'],axis=1,inplace=True)


# In[34]:


df3.head()


# In[35]:


df3.info()


# In[36]:


y=df3['MinUnitPrice']


# In[37]:


df3['size']=df3['size'].astype('float64')


# In[38]:


df3.describe()


# ## Univariate Analysis 

# In[39]:


corr=df3.corr()


# In[40]:


sns.heatmap(corr,annot=True)


# In[41]:


def hist(df,columns):
    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.show()


# In[42]:


df3.iloc[:,0]


# In[43]:


get_ipython().system(' pip install scipy stats')


# In[44]:


from scipy.stats import chi2_contingency


# In[45]:


crosstab = pd.crosstab(df3['organization_type'], df3['products_variant'])
crosstab


# In[46]:


chi2_result=chi2_contingency(crosstab,)
chi2_result[0]


# In[47]:


x=chi2_result[0]
p=chi2_result[1]


# In[48]:


print(f"The p_value is {p} hence we {x} the null hypothesis  with {chi2_result[2]} degrees of freedom ")


# In[49]:


# Check which features to select for categoriacal variables
def chi_cat(df):
    # Select categorical columns
    cat_columns = df.select_dtypes(include='object').columns
    results = []

    # Loop through each pair of categorical columns
    for i in range(len(cat_columns) - 1):
        for j in range(i + 1, len(cat_columns)):
            col1 = cat_columns[i]
            col2 = cat_columns[j]
            
            # Create a crosstab table
            crosstab = pd.crosstab(df[col1], df[col2])
            
            # Perform Chi-Square test
            chi2_result = chi2_contingency(crosstab)
            p_value = chi2_result[1]
            decision = "reject" if p_value < 0.05 else "accept"
            
            # Print the result
            print(f"Testing relationship between '{col1}' and '{col2}':")
            print(f"The p_value is {p_value:.4f} hence we {decision} the null hypothesis with {chi2_result[2]} degrees of freedom.\n")
            
            # Store the results
            results.append((col1, col2, p_value, decision))
    
    return results
        

    


# In[52]:


chi_cat(df3)


# In[50]:


X=df3


# In[51]:


# Remove Outliers
num_cols=X.select_dtypes(include=['int64','float64'])

num_cols


# In[52]:


cat_cols=X.select_dtypes(include=['object'])

cat_cols


# In[53]:


def check_linearity_numcols(df,columns):
    for col in columns:
        plt.figure(figsize=(6,4))
        plt.scatter(x=df[col],y=y)
        plt.xlabel(col)
        plt.ylabel("Total")
        plt.ticklabel_format(style='plain') 
        plt.title(f'Scatterplot for {col} vs Total Price')
        plt.show()
        
        
        
        


# In[54]:


X.shape


# In[55]:


X


# In[56]:


y=df3['MinUnitPrice']
y.shape


# In[57]:


check_linearity_numcols(num_cols,num_cols.columns)


# In[ ]:


sns.pairplot(X)


# In[ ]:





# In[ ]:





# In[ ]:


# def check_linearity_catcols(df,columns,hue)


# In[ ]:


# Removing Outliers
def remove_outliers(df,columns):
    for col in columns:
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df




# In[ ]:


# check_linearity_catcols(cat_cols,columns):


# In[ ]:


# import pandas as pd
# from scipy.stats import zscore
# def remove_outliers(df, columns):
#     z_scores = zscore(df[columns])
#     abs_z_scores = abs(z_scores)
#     filter_outliers = (abs_z_scores < 3).all(axis=1)  # Using Z-score threshold of 3
#     return df[filter_outliers]

# Apply outlier removal to numerical columns while keeping the index intact
df_no_outliers = remove_outliers(X, num_cols.columns)

# Step 3: Now, df_no_outliers contains both cleaned numerical columns and aligned categorical columns
# The categorical columns remain unchanged and are preserved using the same indices
df_cleaned = pd.concat([df_no_outliers[num_cols.columns], df_no_outliers[cat_cols.columns]], axis=1)

# Print cleaned DataFrame

df_cleaned.reset_index(drop=True, inplace=True)
print(df_cleaned)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
# Select numerical columns (int64 and float64)
df_cleaned_num = df_cleaned.select_dtypes(include=['int64', 'float64'])
print(df_cleaned_num)

# Loop through the numerical columns to create boxplots
for column in df_cleaned_num:
    plt.figure(figsize=(10, 6))
    # Create boxplot for each numerical column
    sns.boxplot(data=df_cleaned_num, y=df_cleaned_num[column])
    plt.title(f'Boxplot for {column}')
    plt.show()



# In[ ]:


df_cleaned['Quantities'].value_counts()


# In[ ]:


def check_distribution(df, columns):
    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ticklabel_format(style='plain') 
        plt.ylabel('Density')
        plt.show()
            
        


# In[ ]:


check_distribution(df_cleaned,df_cleaned.columns)


# In[ ]:


import numpy as np

# def log_transform(df, columns):
#     df_transformed = df.copy()
    
#     # Only apply log transformation to numeric columns
#     for col in columns:
#         if df_transformed[col].dtype in ['int64', 'float64']:  # Ensure the column is numeric
#             # Avoid log(0) by adding a small constant (e.g., 1e-10)
#             df_transformed[col] = np.log(df_transformed[col].replace(0, 1e-10))
    
#     return df_transformed

# # Apply the log transformation
# df_transformed = log_transform(df_cleaned, df_cleaned.columns)
# df_transformed

from scipy.stats import boxcox

def box_cox_transform(df, columns):
    """
    Applies Box-Cox transformation to specified columns in a DataFrame.

    Args:
        df: The DataFrame to transform.
        columns: A list of column names to apply the transformation to.

    Returns:
        A DataFrame with the transformed columns.
    """

    df_transformed = df.copy()

    for col in columns:
        if df_transformed[col].dtype in ['int64', 'float64']:
            # Handle negative values or zero values
            if (df_transformed[col] <= 0).any():
                df_transformed[col] += 1  # Shift values to positive range

            # Apply Box-Cox transformation
            df_transformed[col], _ = boxcox(df_transformed[col])

    return df_transformed

# Apply the Box-Cox transformation
df_transformed = box_cox_transform(df_cleaned, df_cleaned.columns)
df_transformed


# In[ ]:


import numpy as np
import pandas as pd

# Assuming df_transformed is your DataFrame after log transformation

# Calculate skewness and kurtosis
skewness = df_transformed.skew()
kurtosis = df_transformed.kurtosis()

# Identify highly skewed columns
highly_skewed_columns = skewness[abs(skewness) > 1]

print("Highly skewed columns after log transformation:")
print(highly_skewed_columns)


# In[ ]:


check_distribution(df_transformed,df_transformed.columns)


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

def scaling_data(df):
    # Separate numeric and categorical columns
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    df_categorical = df.select_dtypes(exclude=['float64', 'int64'])

    # Replace infinity values with a specific value (e.g., -sys.maxsize or another appropriate value)
    df_numeric.replace([np.inf, -np.inf], -sys.maxsize, inplace=True)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply the scaler to the numeric columns
    df_scaled = scaler.fit_transform(df_numeric)

    # Convert the scaled numeric data back into a DataFrame with the original column names
    df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns)

    # Concatenate the scaled numeric data with the original categorical data
    df_final = pd.concat([df_scaled, df_categorical.reset_index(drop=True)], axis=1)

    return df_final

# Example usage:
df_cleaned_scaled = scaling_data(df_transformed)  # Assuming df_cleaned is your DataFrame
df_cleaned_scaled.head()
df_cleaned_scaled.columns


# In[ ]:


import pandas as pd

def cat_cols_binary(df):
    # Separate numeric and categorical columns
    df_num = df.select_dtypes(include=['int64', 'float64'])
    df_cat = df.select_dtypes(include=['object'])

    # Apply get_dummies to categorical columns (one-hot encoding)
    df_cat_encoded = pd.get_dummies(df_cat, drop_first=True)

    # Concatenate the numeric columns (scaled) with the encoded categorical columns
    df_final_new = pd.concat([df_num.reset_index(drop=True), df_cat_encoded.reset_index(drop=True)], axis=1)

    return df_final_new

# Example usage:
# Assuming df_cleaned is your original DataFrame, and df_scaled contains your scaled numeric columns
df_cleaned_cat = cat_cols_binary(df_cleaned_scaled)
df_cleaned_cat.head()
df_cleaned_cat.columns


# In[ ]:





# In[ ]:





# In[ ]:


# Filter both numerical and categorical data based on these indices


# In[ ]:


# df_feature_selection=pd.concat([df_num_cleaned,cat_cols_cleaned],axis=1)
# df_feature_selection


# In[ ]:


# Y_feat=df_feature_selection['MinUnitPrice']
# Y_feat.shape


# In[ ]:


# X_feat=df_feature_selection.drop(['MinUnitPrice'],axis=1)
# X_feat.shape
# print(X_feat.dtypes)


# In[ ]:


# from sklearn.feature_selection import mutual_info_classif
# import matplotlib.pyplot as plt
# %matplotlib inline
# importances=mutual_info_classif(X_feat,Y_feat)
# feat_importance = pd.Series(importances, index=dataframe.columns)

# # Plotting the feature importances
# plt.figure(figsize=(10, 6))
# feat_importance.sort_values(ascending=False).plot(kind='bar')
# plt.title('Feature Importances using Mutual Information')
# plt.ylabel('Importance Score')
# plt.xlabel('Features')
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:








# In[ ]:


y_df=df_cleaned_cat['MinUnitPrice']
y_df.shape


# In[ ]:


X_df = df_cleaned_cat.drop('MinUnitPrice',axis=1)
X_df


# # Feature Selection for Categorical and Numerical Columns

# In[ ]:


# from sklearn.feature_selection import SelectKBest, f_regression

# # Initialize SelectKBest with the f_regression function
# kbest_features = SelectKBest(f_regression, k=3)

# # Fit and transform the data to select the best features
# X_kbestfeatures = kbest_features.fit_transform(X_cat_cols, Y)

# # Print the original and reduced number of features
# print("Original number of features:", X_cat_cols.shape[1])
# print("Reduced number of features:", X_kbestfeatures.shape[1])



# In[ ]:


from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import matplotlib.pyplot as plt

# Assuming X_transformed is your DataFrame with encoded categorical features and numerical features
# And Y is your numerical target variable

# Compute mutual information scores for regression
mi_scores = mutual_info_regression(X, y)

# Create a Pandas Series for feature importances
feat_importance = pd.Series(mi_scores, index=X.columns)

# Get the top 20 features
top_20_features = feat_importance.sort_values(ascending=False).head(20)

# Plotting the top 20 feature importances
plt.figure(figsize=(12, 8))
top_20_features.plot(kind='bar')
plt.title('Top 20 Feature Importances using Mutual Information (Regression)')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.show()



# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression - MSE: {mse_linear}, R^2: {r2_linear}")

# Polynomial Regression (Degree 2 as an example)
poly = PolynomialFeatures(degree=2)
poly_model = make_pipeline(poly, LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression (Degree 2) - MSE: {mse_poly}, R^2: {r2_poly}")

# Choose the model based on MSE and R^2



# In[ ]:


from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=5)  # Adjust alpha as needed
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lasso)
print("Mean Squared Error:", mse)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f' r2_lasso:{r2_lasso}')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming y_test, y_pred_lasso, and y_pred_poly are already defined
# y_pred_linear is used for the Lasso predictions (straight line), y_pred_poly is for polynomial predictions

# Create a DataFrame for plotting
df_test_pred = pd.DataFrame({
    'ytest': y_test,
    'y_pred_lasso': y_pred_lasso,
    'y_pred_poly': y_pred_poly
})

# Plotting
plt.figure(figsize=(10, 8))

# Scatter plot for Lasso predictions
plt.scatter(y_test, y_test, color='black', label='Actual Values', alpha=1, marker='o')
plt.scatter(y_test, y_pred_linear, color='blue',alpha=0.5, label='Linear Predictions')

# Scatter plot for polynomial model predictions
plt.scatter(y_test, y_pred_poly, color='red',alpha=0.5, label='Polynomial Predictions')

# Fit a straight line to the Linear predictions
# Fit line for Linear
coeffs_linear = np.polyfit(y_test, y_pred_linear, 1)
poly_line_linear = np.poly1d(coeffs_linear)
plt.plot(y_test, poly_line_linear(y_test), color='blue',alpha=0.2, linestyle='--', label='Linear Fit Line')

# Fit a polynomial curve to the polynomial model predictions
# Fit curve for Polynomial
coeffs_poly = np.polyfit(y_test, y_pred_poly, 2)  # Change the degree if needed
poly_curve = np.poly1d(coeffs_poly)
plt.plot(y_test, poly_curve(y_test), color='red', alpha=0.2,linestyle='--', label='Polynomial Fit Curve')

plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[ ]:


import statsmodels.api as sm
import pandas as pd

# Assuming your data is already split into X_train and y_train

# 1. Add a constant to the model for the intercept
X_train_const = sm.add_constant(X_train)

# 2. Fit the OLS model
ols_model = sm.OLS(y_train, X_train_const).fit()

# 3. Print the model summary
print(ols_model.summary())



# In[ ]:


from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=3)  # Adjust alpha as needed
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - MSE: {mse_ridge}, R^2: {r2_ridge}")


# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Create the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Fit the model on PCA-transformed training data
xgb_model.fit(X_train, y_train)

# Make predictions on the PCA-transformed test data
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error XGBoost:", mse)
print("R2 Score XGBoost:", r2)


# In[ ]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming X_df and y_df are already scaled

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Train SVR model
regressor = SVR(kernel='rbf')  # Use 'rbf' kernel (default) for non-linear regression
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error SVR: {mse}')
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')


# In[ ]:


import matplotlib.pyplot as plt

# Plotting Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Actual vs Predicted Values')
plt.legend()
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# Assuming X_df and y_df are already scaled

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Build and train the MLP model (Neural Network)
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, activation='relu', solver='adam')

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model with Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error MLPRegressor: {mse}')

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')


# In[ ]:


joblib.dump(scaling_data, 'scaling_data.pkl')


# In[ ]:


import os
print(os.getcwd())


# In[ ]:


# Load the trained model with the correct path
import pandas as pd
import numpy as np
import joblib
import os
os.chdir('C:\\Users\\Wishes Lawrence')
xgb_model = joblib.load('C:\\Users\\Wishes Lawrence\\xgb_model.pkl')
  # This will show your current working directory
scalar=joblib.load('C:\\Users\\Wishes Lawrence\\scaling_data.pkl')



# In[ ]:


import pandas as pd
import numpy as np
import joblib
import os

# Change to the desired directory
os.chdir('C:\\Users\\Wishes Lawrence')

# Load the XGBoost model
xgb_model = joblib.load('xgb_model.pkl')

# Check if the scaler file exists before loading
scaler_path = 'scaling_data.pkl'
if os.path.exists(scaler_path):
    scalar = joblib.load(scaler_path)
else:
    print(f"File {scaler_path} not found. Please check the file path.")

# Display the current working directory
print("Current working directory:", os.getcwd())


# In[ ]:


import pandas as pd
import numpy as np
import joblib

# Sample data structure based on your description
data = {
    'Total': [300000],
    'Quantities': [50000],
    'size': [1.8],
    'organization_type': ['State Government'],
    'buying_mode': ['Direct'],
    'brands': ['LEVRAM'],
    'models': ['HTK3E4-12'],
    'variant': ['Non-Vacuum'],
    'products_variant': ['K3 EDTA']
}

# Convert to DataFrame
new_data_df = pd.DataFrame(data)

### 1. Preprocess Numerical Columns
numerical_cols = ['Total', 'Quantities', 'size']

# Apply the remove_outliers function (assuming you've already defined it)
new_data_df = remove_outliers(new_data_df, numerical_cols)

# Apply scaling to numerical columns (assuming you've already defined scaling_data())
scaled_data = scaling_data(new_data_df[numerical_cols])

# Convert scaled data back to DataFrame
df_num = pd.DataFrame(scaled_data, columns=numerical_cols)

### 2. Preprocess Categorical Columns
categorical_cols = ['organization_type', 'buying_mode', 'brands', 'models', 'variant', 'products_variant']

# One-hot encode the categorical columns
df_cat_cols = pd.get_dummies(new_data_df[categorical_cols])

### 3. Concatenate the Scaled Numerical Data and Encoded Categorical Data
df_concat = pd.concat([df_num, df_cat_cols], axis=1)

### 4. Load the Trained Model and Make Prediction
# Load the trained model
xgb_model = joblib.load('xgb_model.pkl')

# Predict the MinUnitPrice using the preprocessed new data
predicted_min_unit_price = xgb_model.predict(df_concat)

print(f"Predicted MinUnitPrice: {predicted_min_unit_price[0]}")



# In[ ]:


# Check the shape of scaled_data
print(f"Shape of scaled data: {scaled_data.shape}")  # Should print (458, 3) if you expect 3 columns

# Check the numerical columns
print(f"Numerical columns: {numerical_cols}")  # Should print 3 column names if you expect 3

# Ensure the number of columns in scaled_data matches the length of numerical_cols
if scaled_data.shape[1] == len(numerical_cols):
    # Convert scaled data back to DataFrame
    df_num = pd.DataFrame(scaled_data, columns=numerical_cols)
else:
    print(f"Mismatch! Scaled data has {scaled_data.shape[1]} columns but numerical_cols has {len(numerical_cols)} names.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




