# Data-Manipulation-with-Pandas
Tips and tricks when using data manipulation in Python and Pandas

- git remote add origin bitbucket.giturladdress
or - git remote set-url origin
- git pull
- git fetch --all
- git branch
- git checkout feature/sales_predictor

### Install redis-docker
https://kb.objectrocket.com/redis/how-to-install-redis-on-ubuntu-using-docker-505
https://www.youtube.com/watch?v=3muR5gB8x2o&t=402s

### Connect to Google Cloud MYSQL 
https://github.com/GoogleCloudPlatform/getting-started-python/issues/129
```
I experienced this issue when running a flask app locally against a remote CloudSQL db instance (using cloud_sql_proxy). My SQLALCHEMY_DATABASE_URI Connection looked like:

mysql+pymysql://{<user-name}:{<user-password>}@{<db-hostname>}/{<database-name>}?unix_socket=/cloudsql/{<connection-name>}

Turns out connections to CloudSQL can only use either TCP or unix socket, not both. Apparently the proxy uses TCP connection

Solution: remove "unix_socket" param when running locally against the live URI so it looks like this:

mysql+pymysql://{<user-name}:{<user-password>}@{<db-hostname>}/{<database-name>}
```

### Import function from parent folders __init__.py file
https://stackoverflow.com/questions/38955895/import-variable-from-parent-directory-in-python-package

https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html

```
import sys, os.path
sys.path.append(os.path.abspath('../'))
from app import db
```

Replace app with name of parent folder

### Import functions from child folders (model_admins.py in this case, this import is in the __init__.py file)
```
models_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__) + '/models/')))
sys.path.append(models_dir)

from model_admins import VideoFileModelView, FacePhotoModelView, VideoHashListModelView, ScreenshotPhotoModelView # Import models admin pages
```
FILE STRUCTURE
```
app/
├── __init__.py
├── models
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── model_admins.cpython-39.pyc
│   │   └── models.cpython-39.pyc
│   ├── model_admins.py
│   └── models.py
```

### Rearange columns in dataframe
https://stackoverflow.com/questions/35321812/move-column-in-pandas-dataframe/35322540
```
  a  b   x  y
0  1  2   3 -1
1  2  4   6 -2
2  3  6   9 -3
3  4  8  12 -4
```

```
df = df[['a', 'y', 'b', 'x']]
```

### Group by and Sum Pandas
https://stackoverflow.com/questions/39922986/pandas-group-by-and-sum
```
df.groupby(['Fruit','Name']).sum()
```

### Forecast sales
https://www.kaggle.com/cdabakoglu/time-series-forecasting-arima-lstm-prophet
https://www.datacamp.com/community/tutorials/xgboost-in-python

### One-Hot-Encode Pandas dataframe
https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/
```
y = pd.get_dummies(df.Countries, prefix='Country')
print(y.head())
```

### Delete rows from Pandas dataframe based on column value
https://stackoverflow.com/questions/38862587/pandas-dataframe-drop-all-the-rows-based-one-column-value-with-python

```
df[df["name"] != 'tom']

 or 

df[~df['name'].str.contains('tom')]

To remove on multiple criteria  -- "~" is return opposite of True/False

df2[~(df2["name"].isin(['tom','lucy']))]
```

### Run function on all rows in dataframe df.apply()
http://jonathansoma.com/lede/foundations/classes/pandas%20columns%20and%20functions/apply-a-function-to-every-row-in-a-pandas-dataframe/
```

height	width
0	40.0	10
1	20.0	9
2	3.4	4

# Use the height and width to calculate the area
def calculate_area(row):
    return row['height'] * row['width']

rectangles_df.apply(calculate_area, axis=1)


0    400.0
1    180.0
2     13.6
dtype: float64

# Use .apply to save the new column if we'd like
rectangles_df['area'] = rectangles_df.apply(calculate_area, axis=1)

rectangles_df
height	width	area
0	40.0	10	400.0
1	20.0	9	180.0
2	3.4	4	13.6
```

### Create distance matrix between lats and longs pandas dataframe
https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/

### Predict value random forest XG Boost
https://medium.com/@oemer.aslantas/forecasting-sales-units-with-random-forest-regression-on-python-a75d92910b46
https://medium.com/@oemer.aslantas/a-real-world-example-of-predicting-sales-volume-using-xgboost-with-gridsearch-on-a-jupyternotebook-c6587506128d

### Create dataframe based on dataframe index
https://stackoverflow.com/a/53482813/4861086
```
Filter_df  = df[df.index.isin(my_list)]
```

### Replace NaN Values with Zeros in Pandas DataFrame
- For a single column using pandas: df['DataFrame Column'] = df['DataFrame Column'].fillna(0)
- For a single column using numpy: df['DataFrame Column'] = df['DataFrame Column'].replace(np.nan, 0)
- For an entire DataFrame using pandas: df.fillna(0)
- For an entire DataFrame using numpy: df.replace(np.nan,0)

### Unmerge cells and fill in blanks using Excel
https://www.ablebits.com/office-addins-blog/2018/03/07/unmerge-cells-excel/

### Drop all rows after Index Pandas dataframe OR drop rows within a given range Pandas dataframe
```
df.drop(df.iloc[:, 86:], inplace = True, axis=1)  # Drop all columns after the 86th

df.drop(df.index[3:5])  # Drop columns between the 3rd and 5th
```

### Merge the values of 2 rows into a column_title string with a delimeter
```
df.columns = (df.loc[0].astype(str).values + ' - ' + df.loc[1].astype(str).values)
# df = df.reset_index(drop=True)

```

### Convert some columns into rows while leaving the rest the way they are
Before:
```
# Initial DF

	Employee details - Business Unit	Employee details - Full name	2020-07-01 00:00:00 - In	2020-07-01 00:00:00 - Out	2020-07-02 00:00:00 - In	2020-07-02 00:00:00 - Out	2020-07-03 00:00:00 - In	2020-07-03 00:00:00 - Out	2020-07-04 00:00:00 - In	2020-07-04 00:00:00 - Out	...	2020-07-27 00:00:00 - In	2020-07-28 00:00:00 - Out	2020-07-28 00:00:00 - In	2020-07-29 00:00:00 - OUT	2020-07-29 00:00:00 - In	2020-07-30 00:00:00 - Out	2020-07-30 00:00:00 - In	2020-07-31 00:00:00 - Out	2020-07-31 00:00:00 - In	2020-07-31 00:00:00 - Out
2	Distribution	Paul Kang'ethe Kuria	36.5	36.4	37.2	36.4	36.4	36.7	35.1	36.7	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	Commercial	Samson Musyoka	35.7	36.7	37	36.7	36.7	35.7	35.6	36.4	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
4	Deport Clerk	Sylvester Ngesa	36.2	36.7	36	36.7	36.7	36.5	35.9	36.2	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
5	Fullfiller	James Mwendwa	36.7	36.5	36.7	36.5	36.5	36.2	36.6	36.7	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
6	Offloader	Nicholas Kyalo	35.9	36.4	36.2	36.4	36.4	36.6	35.8	36.5	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
5 rows × 60 columns

```
https://stackoverflow.com/a/28654127/4861086
```
# Code Snippet
df1 = df1.melt(id_vars=["Employee details - Business Unit", "Employee details - Full name"], 
        var_name="Date", 
        value_name="Value")
```
After:
```
	Employee details - Business Unit	Employee details - Full name	Date	Value
0	DISTRIBUTION	John Gatere	2020-07-01 00:00:00 - In	35.9
1	DISTRIBUTION	Daniel Musyoka	2020-07-01 00:00:00 - In	34.9
2	Waiyaki Way	Abisai Elia Muthoni	2020-07-01 00:00:00 - In	35.1
```

### Split one pandas column into two different columns based on delimeter
Before
```
	Employee details - Business Unit	Employee details - Full name	Date	Value
0	DISTRIBUTION	John Gatere	2020-07-01 00:00:00 - In	35.9
1	DISTRIBUTION	Daniel Musyoka	2020-07-01 00:00:00 - In	34.9
2	Waiyaki Way	Abisai Elia Muthoni	2020-07-01 00:00:00 - In	35.1
```
https://cmdlinetips.com/2018/11/how-to-split-a-text-column-in-pandas/
```
# Split Column Into 2

df1[['Date','In/Out']] = df1.Date.str.split(" - ",expand=True)
df1.head()
```
After
```
	Employee details - Business Unit	Employee details - Full name	Date	Value	In/Out
0	Distribution	Paul Kang'ethe Kuria	2020-07-01 00:00:00	36.5	In
1	Commercial	Samson Musyoka	2020-07-01 00:00:00	35.7	In
2	Deport Clerk	Sylvester Ngesa	2020-07-01 00:00:00	36.2	In
```

### Create column and fill it in with particular value
https://stackoverflow.com/a/34811984/4861086
```
df['A'] = 'foo'
```

### Split dataframe based on value in one column
```
# Split dataframe based if [in or out] exists in the In/Out column and then concatenate
in_df = df1[df1['In/Out'].str.contains('In', case=False)]
out_df = df1[df1['In/Out'].str.contains('Out', case=False)]
```

### Drop all rows that have NaN as a value in a certain column pandas
https://stackoverflow.com/a/13413845/4861086
```
df = df[df['EPS'].notna()]
```

### Turn distinct column values into column titles pandas
before
```
    key       val
id
2   foo   oranges
2   bar   bananas
2   baz    apples
3   foo    grapes
3   bar     kiwis
```
after
```
key      bar     baz      foo
id                           
2    bananas  apples  oranges
3      kiwis     NaN   grapes
```
https://stackoverflow.com/a/26256360/4861086
```
>>> df.reset_index().groupby(['id', 'key'])['val'].aggregate('first').unstack()
```

### Delete columns that end with certain text pandas
https://stackoverflow.com/a/46346235/4861086
```
df1 = df.loc[:, ~df.columns.str.endswith('Name')]
```

### Drop column whose title contains string pandas
https://stackoverflow.com/a/44272830/4861086
```
df = df[df.columns.drop(list(df.filter(regex='Test')))]
```


### Filter dataframe based on column values pandas
```
higher_df = df[(df.price_per_KG > df.median_market_price)]
lower_df = df[(df.price_per_KG < df.median_market_price)]
equal_df = df[(df.price_per_KG == df.median_market_price)]
```

### Turn Row into column header
https://stackoverflow.com/a/26147330/4861086
```
 df.columns = df.iloc[1]
```

### Access column name while looping through row itertuples
https://stackoverflow.com/a/43620031/4861086
```
for row in df.itertuples():
    print(row.A)
    print(row.Index)
```

### Create calculated column from other columns pandas
```
df['new_col'] = (df.col2/df.col3)
```

### Plot a linegraph Python Pandas
```
import seaborn as sns

# Week Trends
sns.set(rc={'figure.figsize': (19, 8)})
sns.lineplot(df['Week'], df['price_per_KG'], label="Our Price")
sns.lineplot(df['Week'], df['median_market_price'], label="Market Price")
```

### Perform Train-Test Split and build model on it + Feature Importance
- Model building is done after removing unneedded features in *predictors* as shown in the snippet below.
```
#Breaking the data and selecting features , predictors
from sklearn.model_selection import train_test_split
predictors=df_final.drop(['Sold Units','Date'],axis=1)
target=df_final['Sold Units']
x_train,x_cv,y_train,y_cv=train_test_split(predictors,target,test_size=0.2,random_state=7)
```
```
#Hypertuned Model
model = RandomForestRegressor(oob_score = True,n_jobs =3,random_state =7,
                              max_features = "auto", min_samples_leaf =4)

model.fit(x_train,y_train)
```
```
#R2 Score
y_pred = model.predict(x_cv)
r2_score(y_cv, y_pred)
```
```
#Plot feature importance
feat_importances = pd.Series(model.feature_importances_, 
                             index=predictors.columns)
feat_importances.nlargest(10).plot(kind='barh')

```

### Test Multiple Models in One Go (Train-Test split first!!!!)
```
#Import ML Algorithms
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

#Comparing Algorithms
def scores(i):
    lin = i()
    lin.fit(x_train, y_train)
    y_pred=lin.predict(x_cv)
    lin_r= r2_score(y_cv, y_pred)
    s.append(lin_r)
#Checking the scores by using our function
algos=[LinearRegression,KNeighborsRegressor,RandomForestRegressor,Lasso,ElasticNet,DecisionTreeRegressor]
s=[]
for i in algos:
    scores(i)
    
    
#Checking the score
models = pd.DataFrame({
    'Method': ['LinearRegression', 'KNeighborsRegressor', 
              'RandomForestRegressor', 'Lasso','DecisionTreeRegressor'],
    'Score': [s[0],s[1],s[2],s[3],s[4]]})
models.sort_values(by='Score', ascending=False)
```

### Plot Week on Week trends
```
import seaborn as sns
sns.lineplot(df['Week'],df['Sold Units'])

#Yearly Trend
sns.lineplot(df['Year'],df['Sold Units'])
```


### Loop rows in specific columns in 2 dataframes while extracting value from a row and wrting to another Pandas Dataframe
- Loop through 1st dataframe
- initiate variable
- Loop through 2nd dataframe
- if date in 1st df appears in 2nd df, create variable that holds new number
- write new number to 1st dataframe
```
for i, row_df1 in df1.iterrows():
    predicted_sales_volumes = 0
    for i_2, row_df2 in df2.iterrows():
        if row_df1['delivery_date'] in row_df2['all_predicted_date']:
            predicted_sales_volumes = int(predicted_sales_volumes) + int(row_df2['average_delivery_weight'])
    df1.at[i, 'predicted_volumes'] = predicted_sales_volumes
```

### TypeError: argument of type 'Timestamp' is not iterable
Convert the column that is being traversed to a list containing the contents of the row while looping thorugh it
```
for i, row_volumes in df2.iterrows():
    market_price = 0
    market_price_delta = 0
    for i_2, row_market_price in df1.iterrows():
    #important part !!!!!!
        date_list = []
        date_list.append(pd.to_datetime(row_market_price['date']))
        if row_volumes['delivery_date'] in date_list:
	#important part ends !!!!!!
            market_price = int(row_market_price['price_kg'])
            market_price_delta = int(row_volumes['price_per_KG']) - market_price
    df2.at[i, 'median_market_price'] = market_price
    df2.at[i, 'market_price_delta'] = market_price_delta
```


### Pandas error= ValueError: cannot set a Timestamp with a non-timestamp list
The column was a timestamp datatype. Delte the column or make a new one with the same name.

### Delete pandas dataframe row if column has 0
https://stackoverflow.com/a/18173074/4861086
```
df = df[df.line_race != 0]
```


### Create column with  multiple dates between date range
- https://stackoverflow.com/a/39107328/4861086 (Collect dates between datetime ranges)
- https://stackoverflow.com/a/45670296/4861086 (Loop through rows in a dataframe)

- Begin loop through dataframe
- Collect dates as list (by default)
- Extract date from timestamp object
- Write to dataframe

```
for i, row in df.iterrows():
    range_val = pd.date_range(row['earliest_delivery'], row['latest_delivery'], freq=pd.DateOffset(days=row['avgtime_days']))
    range_val = range_val.date
    df.at[i, 'new_predicted_date'] = (range_val)
```

### Extract date from timestamp object
https://stackoverflow.com/a/19106012/4861086
```
In [243]: index = DatetimeIndex(s)

In [244]: index
Out[244]:
<class 'pandas.tseries.index.DatetimeIndex'>
[2013-10-01 00:24:16, 2013-10-02 00:24:16]
Length: 2, Freq: None, Timezone: None

In [246]: index.date
Out[246]:
array([datetime.date(2013, 10, 1), datetime.date(2013, 10, 2)], dtype=object)
```
### ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
You are refrencing a series instead of an individual value, replace:
```
for row in next_purchase.itertuples():
	earliest_date = row.earliest_delivery
```
with
```
for row in next_purchase.itertuples():
	row.latest_delivery
	
	or
	
latest_delivery = next_purchase.at[row.Index, 'latest_delivery']
```
or use a mask with datetime
### Get average difference between dates SQL
https://stackoverflow.com/a/32723609/4861086
```
SELECT id_usuarioweb, CASE 
        WHEN COUNT(*) < 2
            THEN 0
        ELSE DATEDIFF(dd, 
                MIN(
                    dt_fechaventa
                ), MAX(
                    dt_fechaventa
                )) / (
                COUNT(*) - 
                1
                )
        END AS avgtime_days
FROM mytable
GROUP BY id_usuarioweb
```

### Convert pandas column to numbers
https://stackoverflow.com/a/28648923/4861086
```
# convert all columns of DataFrame
df = df.apply(pd.to_numeric) # convert all columns of DataFrame

# convert just columns "a" and "b"
df[["a", "b"]] = df[["a", "b"]].apply(pd.to_numeric)
```

### Round off pandas column
https://stackoverflow.com/questions/26133538/round-a-single-column-in-pandas
```
df.value1 = df.value1.round()
```

### Add column with numbers to datime column pandas
https://stackoverflow.com/a/46907838/4861086
```
df['new'] = df['transaction_date'] + pd.to_timedelta(df['payment_plan_days'], unit='d')
```

### Add a number of days to column with date
https://stackoverflow.com/a/46571728/4861086
```
df['x_DATE'] = df['DATE'] + pd.DateOffset(days=180)
```

### Pandas Convert negative column to positive
```
next_purchase['avgtime_days'] = next_purchase['avgtime_days'].abs()
```

### Add aggregate function to one of the where clauses  SQL
https://stackoverflow.com/a/19828119/4861086 (In the comment)

### Sum distinct values in Pandas Dataframe columns after group by 
- Group by all required items plus columns we want to sum their distinct values.
- Do a scond group by where you sum the values in the column with distinct values.


### Get all data from dataframe that is in a list/ Get all data that has nothing in list pandas
Get all data that is not in the list
```
new_df = (old_df[~old_df.column_name.isin(list_name)])
```
Get all the data that is in a list
```
new_df = (old_df[old_df.column_name.isin(list_name)])
```

### Create/Add Column in pandas dataframe based on other dataframe
- Make sure *other* dataframe only has the columns we need to add and the columns we will merge with
- Perform an inner merge on both with origial df on left and other df on right
```
orig_df = orig_df.merge(other_df, how='inner', left_on=['delivery_date', 'product_item_name'], 
right_on=['sale_date', 'product_item_name'])

volumes_sold = volumes_sold.drop(['delta'], axis=1)
```


### Create/Add Column values based on date/time period
```
# Create Seasonality Feature

# Create mask for different season time periods
season_1_start = '2019-09-01'
season_1_end = '2020-01-31'

season_2_start = '2020-04-01'
season_2_end = '2020-07-30'

season1_mask = ((volumes_sold['df'] >= season_1_start) & (volumes_sold['df'] <= season_1_end))
season2_mask = ((volumes_sold['df'] >= season_2_start) & (volumes_sold['df'] <= season_2_end))

conditions = [
    (season1_mask == True),
    (season1_mask == False),
    (season2_mask == True),
    (season2_mask == False)
]

choices = [1,0,1,0]

df['in_season'] = np.select(conditions, choices, default=0)
```

### Sum many/all columns in dataframe
- Split dataframe by deleting columns we dont want added (https://stackoverflow.com/a/34683105/4861086)
- Perfrom groupping on columns we dont want added
- Perform add on columns we want added (https://stackoverflow.com/questions/35001996/pandas-grouping-dataframe-by-hundreds)maybe
```
# Copy Columns we need
new = old[['A', 'C', 'D']].copy()

# Delete Copied Columns
old = old.drop(columns=[ 'A', 'C', 'D'])

# Group many columns at once in new df
volumes_sold_encoded = volumes_sold_encoded.groupby('delivery_date').sum()

volumes_sold_shop_type.drop(columns=['Unnamed: 0'])

volumes_sold_encoded = volumes_sold_encoded.reset_index()
# Group old DF

# Merge both without suffix
volumes_sold_result_df.merge(volumes_sold_encoded, left_on='delivery_date', right_on='delivery_date', suffixes=(False, False))

```

### Create Pandas columns based on element in list
https://stackoverflow.com/questions/47893355/check-if-value-from-a-dataframe-column-is-in-a-list-python
- Set conditions(Check if value is in list) or not (https://stackoverflow.com/questions/14057007/remove-rows-not-isinx)

- Create choices based on whether or not item is in list

- Apply choices using np.select (https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column)
```
conditions = [
    (volumes_sold['delivery_date'].isin(kenyan_holidays)),  # If item is in list
    (~volumes_sold['delivery_date'].isin(kenyan_holidays))] # If item is not in list

choices = [1,0] # Apply 1st item if delivery date in list, 2nd item if item not in list

volumes_sold['holiday'] = np.select(conditions, choices, default=0) # Perform Operation
```


### Plot Regression when LogTransformed
https://stackoverflow.com/a/51061094/4861086
```
# sns.pairplot(df,kind='reg',height=8, x_vars=['column1'], y_vars=['column2'])
g = sns.jointplot( "column1", "column2", data=df,
                  kind="reg", truncate=False,
                  color="m", height=7, logx = True)

g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')

plt.title('Watermelons ', y=20, fontsize = 16)
```
### Get reorder rate from loan dataset
-group by customer, 
-then product name 
-then count the number of deliveries
-and distinct number of weeks served, total deliveries for number of weeks

2.3  Get weekly reorder rate for each product per customer
```
deliveries_financed_loans['Week_Number'] = deliveries_financed_loans['delivery_date'].dt.week

reorder_rate_df = deliveries_financed_loans.groupby(['Unique_Stalls', 'product_name']).agg(
    number_of_deliveries_per_customer=('delivery_callback_id', np.count_nonzero),
    distinct_number_of_weeks=('Week_Number', pd.Series.nunique)
)

reorder_rate_df['weekly_reorder_rate'] = reorder_rate_df['number_of_deliveries_per_customer']/reorder_rate_df['distinct_number_of_weeks']

reorder_rate_df.to_excel("reorder_rate_df.xlsx") 

new_reorder_rate_df = reorder_rate_df.groupby(['product_name']).agg(
    average_weekly_customer_reorder_rate=('weekly_reorder_rate', np.average)
)

 new_reorder_rate_df.reset_index(inplace=True)
```
2.4  Filter to only products that we focus on in analysis
```
# Drop via logic: similar to SQL 'WHERE' clause

product_list = ['Ajab home baking flour','Pembe Maize Flour','Soko Maize Flour','Biryani Rice',
 'Halisi Cooking Oil','Postman Cooking Oil','Kabras Sugar','Afia Mango',
 'Salit Cooking Oil','Pembe Home Baking Flour','Bananas','Potatoes',
 'Tomatoes','Onions','Watermelon']

# new_reorder_rate_df[~new_reorder_rate_df.isin(product_list)]


new_reorder_rate_df = new_reorder_rate_df[new_reorder_rate_df.product_name.isin(product_list)]

```


### Count distinct Pandas
https://stackoverflow.com/questions/15411158/pandas-countdistinct-equivalent
```
table.groupby('YEARMONTH').CLIENTCODE.nunique()
```

### Convert Pandas Column COntent to Python List
https://stackoverflow.com/a/22341390
```
col_one_list = df['one'].tolist()
```

### Delete/Drop rows based on value Pandas dataframe
https://stackoverflow.com/questions/41934584/how-to-drop-rows-by-list-in-pandas
https://hackersandslackers.com/pandas-dataframe-drop/
```
print (df[~df.column_name.isin(list_name)])
```

### Box-Plot to find outliers in variables Plot-Ly Pandas Python
https://plotly.com/python/box-plots/
```
import plotly.express as px

fig = px.box(df, y="column_1")
fig.show()
```

### Get customers that bought product in one month and not the next month
https://stackoverflow.com/a/47107164/4861086

Collect or create datetime data that we will use to compare the two different values
```
feb_date = '2020-02-01'
march_date = '2020-03-01'
april_date = '2020-04-01'

feb_date = pd.to_datetime(feb_date)
march_date = pd.to_datetime(march_date)
april_date = pd.to_datetime(april_date)

df['delivery_date'] =  pd.to_datetime(df['delivery_date'])
```

Specify the date periods with which we will be comparing values
```
feb_df = df[(df.delivery_date < march_date) & (df.delivery_date > feb_date)]
march_df = df[(df.delivery_date < april_date) & (df.delivery_date > march_date)] 
```

Merge both DFs
```
df_all = feb_df.merge(march_df.drop_duplicates(), on=['Unique_Stalls_x'], how='left', indicator=True)

#Drop rows that appearedn both
df_all.drop(df_all[df_all._merge == 'both'].index, inplace=True)

# Drop columns wth everything missing
df_all.dropna(axis='columns',how='all')

df_no_longer = df_all.groupby(['Unique_Stalls_x']).agg(
    bales_bought=('uom_count_x', sum)
    )

df_no_longer = df_no_longer.reset_index()
```
### Add distinct count column to dataframe
https://stackoverflow.com/questions/15411158/pandas-countdistinct-equivalent

```
# Number of unique cutomers in a day
new_df = df.groupby('delivery_date').Unique_Stalls.nunique()

#Merge with orignal df
result_df = pd.merge(df,
                 new_df,
                 on='delivery_date',
                 how='left')
		 
result_df['delivery_date']= pd.to_datetime(result_df['delivery_date']) 
```

### Create a correlation plot Pandas Pearsons Coefficient
```
# vendor_drops.fillna(0, inplace = True, axis=0)
corr_df = df.corr()

plt.figure(figsize = (13,10))
sns.heatmap(corr_df, annot=True)
plt.savefig('df_heatmap.png')
```

### Recover deleted file from WSL
https://stackoverflow.com/questions/38819322/how-to-recover-deleted-ipython-notebooks

### Loop thorugh list of dictionary of dictionaries 
https://stackoverflow.com/questions/45592268/python-access-dictionary-inside-list-of-a-dictionary
```
my_nested_dictionary = {'mydict': {'A': 'Letter A', 'B': 'Letter C', 'C': 'Letter C'}}
print(my_nested_dictionary['mydict']['A'])


for key in geocode_result: #list
	for k, v in key.items(): #JSON object collect value
	    if isinstance(v, dict):
		if k == 'geometry': # If the key is geometry, get specified item
		    loc_list.at[item.Index, 'latitude'] = v['location']['lat']
		    loc_list.at[item.Index, 'longitude'] = v['location']['lng']
```

### Concatenate dataframes Pandas (Must have same column names)
```
display('new_banana_drops', 'new_vendor_drops', pd.concat([new_banana_drops, new_vendor_drops]))

new_banana_df = pd.concat([new_banana_drops, new_vendor_drops])
```

### Convert text area names to longitude and latitudes using google maps API Pandas Python

```
%pip install -U googlemaps
import googlemaps

gmaps = googlemaps.Client(key='my_key')

# Geocoding an address
geocode_result = gmaps.geocode('KICC, Nairobi, Kenya')

print (geocode_result)

# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions("Sydney Town Hall",
                                     "Parramatta, NSW",
                                     mode="transit",
                                     departure_time=now)
```
### Drop columns that have all NaNs
```
# Drop columns wth everything missing
df_all.dropna(axis='columns',how='all')
```


### Infinity produced when calculating average (INF)
- Remove all 0s from a column that is undergoing calculations

```
# Remove 0s from dataframe
df = df[(df != 0).all(1)]
```

### Box Cox Power Transform Dataframe Pandas
https://stackoverflow.com/a/22889503/4861086
```
from scipy import stats

# new_banana_df['average_daily_selling_price'] = stats.boxcox(new_banana_df.average_daily_selling_price)[0]
new_banana_df['average_daily_kg_selling_price'] = stats.boxcox(new_banana_df.average_daily_kg_selling_price)[0]
# new_banana_df['volumes_sold_KG'] = stats.boxcox(new_banana_df.volumes_sold_KG)[0]
```

### Pandas error: Truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()
https://stackoverflow.com/questions/36921951/truth-value-of-a-series-is-ambiguous-use-a-empty-a-bool-a-item-a-any-o

Replace *and* or *or* with *&* and *|* respectively. This typically happens when searching with multiple operands.

```
 result = result[(result['var']>0.25) or (result['var']<-0.25)]
 
 result = result[(result['var']>0.25) and (result['var']<-0.25)]
```



### Delete Pandas row based on conditon dataframe
```
df.drop(df[df.score < 50].index, inplace=True)

df = df.drop(df[(df.score < 50) & (df.score > 20)].index)
```

### Log Transform Pandas Feature
```
new_banana_df['volumes_sold_KG'] = np.log(new_banana_df['volumes_sold_KG'])
```

### Add calculated column Pandas Dataframe

```
# Create new column
df['new_clolumn] = ''

# Loop through dataframe appending to calculated column
for row in df.itertuples():
    banana_dr_quantity = df.at[row.Index, 'quantity']
    banana_dr_amount = df.at[row.Index, 'amount']
    df.at[row.Index, 'price_per_KG'] = (banana_dr_amount/banana_dr_quantity)
```


### Provide list or concatenation of items while aggregating group by Pandas
https://stackoverflow.com/a/27360130

```
df = df.groupby(['column_1', 'column_2']).agg(
    delivery_items=('column_3', list)
    )

df = df.reset_index()

##############  or  ##########

df = df.groupby(['column_1', 'column_2']).agg(
    delivery_items=('column_3', sum)
    )

df = df.reset_index()
```


### Count number of occurences of items in Pandas dataframe and append with other data
https://stackoverflow.com/a/55828762
```
df_1 = df.groupby(['product_name']).size()
df_1 = df_1.reset_index()

df = df_1.merge(df, left_on='product_name', right_on='product_name')
```


### Filter for time or date when using a datetime column
https://stackoverflow.com/questions/40192704/filter-pandas-dataframe-for-past-x-days

```
import datetime
import pandas as pd 


df = df[(df.delivery_date < df.loan_startdate) & (df.delivery_date > (pd.to_datetime(df.loan_startdate) - pd.to_timedelta("30day")))]
```


### Count number of weekdays in a week 
https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.busday_count.html
```
>>> # Number of weekdays in January 2011
... np.busday_count('2011-01', '2011-02')
21
>>> # Number of weekdays in 2011
...  np.busday_count('2011', '2012')
260
>>> # Number of Saturdays in 2011
... np.busday_count('2011', '2012', weekmask='Sat')
53
```


### Create new empty column from each value in row Pandas Dataframe
```
for row in df.itertuples():
    print(row.column_name)
    df[row.column_name] = ""
```


### Get number of distinct weeks over whoch something occured weeks
https://stackoverflow.com/questions/31181295/converting-a-pandas-date-to-week-number
- Get the number of weeks indivdually from which something occured
```
df['Week_Number'] = df['delivery_date'].dt.week
```
- Aggregate the week number distinctly when grouping by
```
df = df.groupby(['Unique_Stalls']).agg(
    distinct_number_of_weeks=('Week_Number', pd.Series.nunique),
    distinct_deliveries=('delivery_id', pd.Series.nunique)
    )
    
df = df.reset_index()
```

### None of [Index([..], dtype='object')] are in the [columns]”
https://stackoverflow.com/questions/55652574/how-to-solve-keyerror-unone-of-index-dtype-object-are-in-the-colum
- There is a space in the title of one of our columns


### Error: Pandas unstack problems: ValueError: Index contains duplicate entries, cannot reshape

- Create MultiIndex Dataframe
- Or group by again
- Reset the index
- Try get the Pivot again


### Select rows whose column equals a certain value
https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values
```
df.loc[df['column_name'] == some_value]
```


### Create new dataframe based on certain row values
https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values
```
df = df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]

df = deliveries_df.loc[deliveries_df['delivery_date'] < deliveries_df['loan_startdate']]

```

### Sum everything in a row
https://stackoverflow.com/a/25748826
```
df['e'] = df.sum(axis=1)
```

### Rename Column Pandas
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
```
df.rename(columns={"A": "a", "B": "c"})
```

### Top 10 items based on column value Pandas 
https://cmdlinetips.com/2019/03/how-to-select-top-n-rows-with-the-largest-values-in-a-columns-in-pandas/
```
df.nsmallest(3,'column1')

df.nlargest(10,'column1')
```

### Convert DateTime Column to Days of the Week
https://stackoverflow.com/questions/30222533/create-a-day-of-week-column-in-a-pandas-dataframe-using-python
```
df['day_of_week'] = df['my_dates'].dt.day_name()
```

### Build OLS Linear Regression
```
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')
# lm = smf.ols('np.log(column1) ~ np.log(column2)', data=df).fit()
lm = smf.ols('column1  ~ column2', data=df).fit()

lm.summary()
```

### Create custom Maps with CSV Google
https://webapps.stackexchange.com/a/102780

### Find all rows that match criteria and create new dataframe from it Pandas, Python
https://stackoverflow.com/questions/51004029/create-a-new-dataframe-based-on-rows-with-a-certain-value
```
new_df = old_df[old_df.column_name == 'Biryani Rice']
milk_vendor_drops.head(5)
```

### Delete Pandas dataframe column
https://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe
```
del df['column_name']

df.drop(df.columns[22:56], axis=1, inplace=True)

df_all.drop(['delivery_date_x', 'depot_name_x', 'route_name_x', 'shop_type_x', 'delivery_id_x', 'product_name_x', 
'product_item_name_x', 'Weight_x', 'Amount_x'], axis=1, inplace=True)
```

### Plot Lineatr Regression Seaborn Pandas
https://seaborn.pydata.org/examples/regression_marginals.html
```
import seaborn as sns

sns.jointplot("average_selling_price", "volumes_sold", data=milk_vendor_drops,
                  kind="reg", truncate=False,
                  color="m", height=7)
plt.title('Basmati Rice', y=5, fontsize = 16)
```

### Append values to dataframe column while looping through it / Amend values in dataframe while looping 
https://stackoverflow.com/a/47604317/4861086

```
for row in df.itertuples():
    if <something>:
        df.at[row.Index, 'column_name'] = x
    else:
        df.at[row.Index, 'column_name'] = x

    df.loc[row.Index, 'ifor'] = x
    
for row in df.itertuples():
    if row.column_name:
```

### Convert Pandas Column to datetime
https://stackoverflow.com/a/34507381
```
pd.to_datetime(df.col2, errors='coerce')
```

### Check datatype python pandas

```
if isinstance(VARIABLE, float):
```

### Loop through each row in dataframe pandas
https://stackoverflow.com/a/45716191/4861086
```
for item in df.itertuples():
    print(item.column_name, item.column_name)

```

### Drop Multiple columns in dataframe pandas
https://cmdlinetips.com/2018/04/how-to-drop-one-or-more-columns-in-pandas-dataframe/
```
# pandas drop columns using list of column names
gapminder_ocean.drop(['pop', 'gdpPercap', 'continent'], axis=1)
```

### Aggregrate multiple rows by date Pandas
https://stackoverflow.com/questions/50569899/pandas-how-to-group-by-multiple-columns-and-perform-different-aggregations-on-m
```
banana_drops = agg_banana_df.groupby(['delivery_date']).agg(
    total_volumes_sold=('Weight', sum),
    avg_drop_size=('dropsize', np.average),
    median_drop_size=('dropsize', np.median),
    number_of_unique_customers=('Unique_Stalls', pd.Series.nunique),
    number_of_drops=('number_of_drops', np.average)
    )
    
banana_drops = banana_drops.reset_index()
```
### Draw a heatmap SNS seaborn pandas python
```
corr_df = vendor_drops.corr()
corr_df.to_excel('3_FFV_matrix.xlsx')

plt.figure(figsize = (13,10))
sns.heatmap(corr_df, annot=True)
plt.savefig('ajab_corr_heatmap.png')
```

### Plot linear regression correlation between 2 dfferent values
```
sns.pairplot(vendor_drops, kind='reg', height=10, x_vars=['selling_price'], y_vars=['gross_profit'])
```
OR
```
plt.figure(figsize = (10,7))
sns.regplot(data = vendor_drops , x=vendor_drops['selling_price'], y=vendor_drops['gross_profit'])
```
### Get dataframe correlation coefficent Pandas DataFrame
```
import seaborn as sns

corr_df = new_banana_df.corr()

plt.figure(figsize = (13,10))
sns.heatmap(corr_df, annot=True)
plt.savefig('banana_heatmap.png')
```


### Export dataset pandas
```
new_df.to_excel("ajab_bananas.xlsx")  
```

### Build a Price Elasticity model

https://datafai.com/2017/11/30/price-elasticity-of-demand/
https://www.statworx.com/ch/blog/food-for-regression-using-sales-data-to-identify-price-elasticity/
https://medium.com/teconomics-blog/how-to-get-the-price-right-9fda84a33fe5


### Merge dataframes
https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/
```
(df1)
 	product_name 	number_of_purchases
0 	Afia Mango 	180
1 	Afia Mixed Fruit 	107
2 	Afia Multi-Vitamin 	15
3 	Afia Orange 	4
4 	Afia Tropical 	3

(df2)
	product_name 	total_weight_per_product 	total_amount_per_product
0 	Afia Mango 	872.4 	102950.0
1 	Afia Mixed Fruit 	520.8 	61620.0
2 	Afia Multi-Vitamin 	73.2 	8500.0
3 	Afia Orange 	14.4 	1790.0
4 	Afia Tropical 	21.6 	2420.0


result_df = df_1.merge(df_2, left_on='product_name', right_on='product_name')
result_df.head(5)


	product_name 	number_of_purchases 	total_weight_per_product 	total_amount_per_product
0 	Afia Mango 	180 	872.4 	102950.0
1 	Afia Mixed Fruit 	107 	520.8 	61620.0
2 	Afia Multi-Vitamin 	15 	73.2 	8500.0
3 	Afia Orange 	4 	14.4 	1790.0
4 	Afia Tropical 	3 	21.6 	2420.0

```

### Calculate number of occurences before aggregating

```
### Create column with number of something
number_of_drops_column = banana_drops.groupby(['delivery_date']).size().to_frame(
    name='number_of_drops')
 
### Merge column with the rest of the dataset
agg_banana_df = banana_drops.merge(number_of_drops_column,   
                                     left_on=['delivery_date'] ,right_on=['delivery_date'])

agg_banana_df.head(5)
```

### Get last n commands run jupyter notebook
```
_ih[-10:]
```

### Pandas Group By Aggregate on distinct count or unique count

https://stackoverflow.com/questions/18554920/pandas-aggregate-count-distinct
```
banana_drops = agg_banana_df.groupby(['delivery_date', 'order_date']).agg(
    number_of_unique_customers=('Unique_Stalls', pd.Series.nunique)
    )
    
banana_drops = banana_drops.reset_index()

```
### Replace NaN with 0s
```
DataFrame.fillna()
```

### Create new blank dataframe
https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
```
# Create new column
df['new_clolumn] = ''

# Loop through dataframe appending to calculated column
for row in df.itertuples():
    banana_dr_quantity = df.at[row.Index, 'quantity']
    banana_dr_amount = df.at[row.Index, 'amount']
    df.at[row.Index, 'price_per_KG'] = (banana_dr_amount/banana_dr_quantity)
```

### Plot correlation heatmap pandas python 

```
corr_df = dataframe.corr()
corr_df.to_excel('all_banana_matrix.xlsx')

plt.figure(figsize = (13,10))
sns.heatmap(corr_df, annot=True)
plt.savefig('banana_matrix.xlsx_corr_heatmap.png')
```

### Check for specific values between given dates
```
# Create dataframes with data from the separate date ranges
start_date_old = '2019-11-01'
end_date_old = '2019-11-30'

start_date = '2020-02-01'
end_date = '2020-02-29'

mask_old = (final_banana_df['delivery_date'] >= start_date_old) & (final_banana_df['delivery_date'] <= end_date_old)
old_df = final_banana_df.loc[mask_old]

mask_new = (final_banana_df['delivery_date'] >= start_date) & (final_banana_df['delivery_date'] <= end_date)#
new_df = final_banana_df.loc[mask_new]

# Check if values in one dataframe appear in another and create a new column to hold this boolean
new_df = new_df.assign(in_old_df=new_df.Unique_Stalls.isin(old_df.Unique_Stalls).astype(str))  
new_df = new_df[new_df.in_old_df == 'True']

# Delete column if need be
del new_df['in_old_df']

```

### Pivot Pandas dataframe

```
    foo   bar  baz  zoo
0   one   A    1    x
1   one   B    2    y
2   one   C    3    z
3   two   A    4    q
4   two   B    5    w
5   two   C    6    t


df.pivot(index='foo', columns='bar', values='baz')

bar  A   B   C
foo
one  1   2   3
two  4   5   6
```
- When there is a Multi-Indexed dataframe, reset the index first with
```
df = df.reset_index()
```

### Fill in Nan with 0 / Fill in all blank cells with value pandas
```
df.fillna(0)
```

### Append 1 dataframe to the bottom of another
```
final_df = final_df.append(full_df)
```

### Delete Column if 1st row has a value

https://stackoverflow.com/questions/42377344/drop-multiple-columns-based-on-different-values-in-first-row-of-dataframe
```
mask = df.iloc[0].isin(['Apples','Pears'])
```
```
print (mask)
Fav-fruit     True
Unnamed1     False
Unnamed2      True
Cost         False
Purchsd?     False
Unnamed3     False
Name: 0, dtype: bool
```
```
print (~mask)
Fav-fruit    False
Unnamed1      True
Unnamed2     False
Cost          True
Purchsd?      True
Unnamed3      True
Name: 0, dtype: bool
```
```
print (df.loc[:, ~mask])
```
```
  Unnamed1  Cost Purchsd? Unnamed3
0  Bananas   NaN      Yes       No
1      NaN   0.1      NaN       No
2      NaN   0.3      NaN       No
3      NaN   0.1      Yes      NaN
```

### Insert new row to pandas DataFrame
https://pythonexamples.org/pandas-dataframe-add-append-row/
```
import pandas as pd

data = {'name': ['Somu', 'Kiku', 'Amol', 'Lini'],
	'physics': [68, 74, 77, 78],
	'chemistry': [84, 56, 73, 69],
	'algebra': [78, 88, 82, 87]}

	
#create dataframe
df_marks = pd.DataFrame(data)

new_row = {'name':'Geo', 'physics':87, 'chemistry':92, 'algebra':97}
#append row to the dataframe
df_marks = df_marks.append(new_row, ignore_index=True)
```

### Collect data recieved from a request flask
https://stackoverflow.com/questions/10434599/get-the-data-received-in-a-flask-request

```
# REST Handler
@app.route('/recommend', methods=['POST'])
def collect_test_results():
    if request.method == 'POST':
        Student_Name = request.values.getlist('Student_Name') # Name of the student
        Video_Name = request.values.getlist('Video_Name') # Name of the video
        Is_correct = request.values.getlist('Is_correct') # Whether or not the video is correct
```

### Import file that is in another folder python
https://stackoverflow.com/a/46569406/4861086

```
Since the application folder structure is fixed, we can use os.path to get the full path of the module we wish to import. For example, if this is the structure:

/home/me/application/app2/some_folder/vanilla.py
/home/me/application/app2/another_folder/mango.py
And let's say that you want to import the mango module. You could do the following in vanilla.py:

import sys, os.path
mango_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/another_folder/')
sys.path.append(mango_dir)
import mango
```

OR FROM JUST A SUB-FOLDER 
```
models_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__) + '/models/')))
sys.path.append(models_dir)
import models # models.py
```
