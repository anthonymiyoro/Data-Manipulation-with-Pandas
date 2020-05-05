# pandasTipsAndTricks
Tips and tricks when using data manipulation in Python and Pandas

### Plot Regression when LogTransformed
https://stackoverflow.com/a/51061094/4861086
```
# sns.pairplot(milk_vendor_drops,kind='reg',height=8, x_vars=['average_selling_price'], y_vars=['volumes_sold'])
g = sns.jointplot( "price_per_kg", "number_of_kgs_sold", data=vendor_drops,
                  kind="reg", truncate=False,
                  color="m", height=7, logx = True)

g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')

plt.title('Watermelons ', y=20, fontsize = 16)
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

### Fill in Nan with 0
```
df.fillna(0)
```
