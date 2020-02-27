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
