# pandasTipsAndTricks
Tips and tricks when working with pandas

### SQL Basics
https://blog.hubspot.com/marketing/sql-tutorial-introduction

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
