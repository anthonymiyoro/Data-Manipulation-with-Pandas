# pandasTipsAndTricks
Tips and tricks when working with pandas

### SQL Basics
https://blog.hubspot.com/marketing/sql-tutorial-introduction

```
SELECT
  *
FROM `twigadms.dmslive.cache_finance_deliveries`
WHERE
  product_name in(
    'Ajab home baking flour',
    'Biryani Rice',
    'Dairy Top UHT Milk',
    'Watermelon',
    'Tomatoes',
    'Bananas',
    'Potatoes'
  )
  AND delivery_date BETWEEN '2019-11-01'
  AND '2020-02-15'
LIMIT
  10000;
```

### Append values to dataframe column while looping through it
https://stackoverflow.com/a/47604317/4861086

```
for row in df.itertuples():
    if <something>:
        df.at[row.Index, 'ifor'] = x
    else:
        df.at[row.Index, 'ifor'] = x

    df.loc[row.Index, 'ifor'] = x
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
    print(item.a, item.b)

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
# Aggregrate by delivery date, product name and its SKU, get the median selling price and gross profit
aggregation = {'selling_price':  'median', 'Weight': 'sum', 'gross_profit':  'median'}
new_df = vendor_drops.groupby(['delivery_date','product_name','product_item_name'], as_index=False).agg(aggregation)
```
### Draw a heatmap SNS seaborn pandas python
```
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

### Export dataset pandas
```
new_df.to_excel("ajab_bananas.xlsx")  
```

### Build a Price Elasticity model

https://datafai.com/2017/11/30/price-elasticity-of-demand/

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

### Pandas Group By Aggregate on distinct count

https://stackoverflow.com/questions/18554920/pandas-aggregate-count-distinct
```
banana_drops = agg_banana_df.groupby(['delivery_date']).agg(
    total_volumes_sold=('Weight', sum),
    avg_drop_size=('dropsize', np.average),
    median_drop_size=('dropsize', np.median),
    number_of_unique_customers=('Unique_Stalls', pd.Series.nunique),
    number_of_drops=('number_of_drops', np.average)
    )
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


