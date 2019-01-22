# This was written to solve a Kaggle challenge for an academic project. 
import numpy as np
import pandas as pd
import os
print(os.listdir())
items_df = pd.read_csv('items.csv')
shops_df = pd.read_csv('shops.csv')
cat_df = pd.read_csv('item_categories.csv')
train_df = pd.read_csv('sales_train_v2.csv')

train_df = train_df[train_df['item_price']<100000]
train_df = train_df[train_df['item_cnt_day']<1001]

cat_df.head()

cat_list = list(cat_df.item_category_name)

for i in range(1,8):
    cat_list[i] = 'Access'

for i in range(10,18):
    cat_list[i] = 'Consoles'

for i in range(18,25):
    cat_list[i] = 'Consoles Games'

for i in range(26,28):
    cat_list[i] = 'phone games'

for i in range(28,32):
    cat_list[i] = 'CD games'

for i in range(32,37):
    cat_list[i] = 'Card'

for i in range(37,43):
    cat_list[i] = 'Movie'

for i in range(43,55):
    cat_list[i] = 'Books'

for i in range(55,61):
    cat_list[i] = 'Music'

for i in range(61,73):
    cat_list[i] = 'Gifts'

for i in range(73,79):
    cat_list[i] = 'Soft'

cat_df['cats'] = cat_list
cat_df.head()

train_df.head()

train_df['date'] = pd.to_datetime(train_df.date, format = "%d.%m.%Y")
train_df.head()

pivot_df = train_df.pivot_table(index=['shop_id','item_id'], columns = 'date_block_num', values = 'item_cnt_day', aggfunc='sum').fillna(0.0)
pivot_df.head()

train_cleaned_df = pivot_df.reset_index()
train_cleaned_df['shop_id'] = train_cleaned_df['shop_id'].astype('str')
train_cleaned_df['item_id'] = train_cleaned_df['item_id'].astype('str')

item_to_cat_df = items_df.merge(cat_df[['item_category_id','cats']], 
                                how="inner", 
                                on="item_category_id")[['item_id','cats']]
item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')

train_cleaned_df = train_cleaned_df.merge(item_to_cat_df, how="inner", on="item_id")
train_cleaned_df.head()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
train_cleaned_df['cats'] = le.fit_transform(train_cleaned_df['cats'])
train_cleaned_df = train_cleaned_df[['shop_id','item_id','cats'] + list(range(34))]

train_cleaned_df.head()

X_train = train_cleaned_df.iloc[:, :-1].values
print(X_train.shape)
X_train[:2]

y_train = train_cleaned_df.iloc[:,-1].values

# import xgboost as xgb
# param = {'max_depth':6,  # originally 10
#          'subsample':0.85,  # 1
#          'min_child_weight':0.5,  # 0.5
#          'eta':0.25,
#          'num_round':1000, 
#          'seed':0,  # 1
#          'silent':0,
#          'eval_metric':'rmse',
#          'early_stopping_rounds':100,
#         }
# progress = dict()
# xgbtrain = xgb.DMatrix(X_train, y_train)
# watchlist  = [(xgbtrain,'train-rmse')]
# bst = xgb.train(param, xgbtrain)

from xgboost import XGBRegressor
model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=100, 
    colsample_bytree=0.8, 
    subsample=0.9, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train)], 
    verbose=True, 
    early_stopping_rounds = 10)

from sklearn.metrics import mean_squared_error
preds = bst.predict(xgb.DMatrix(X_train))

rmse = np.sqrt(mean_squared_error(preds, y_train))
print(rmse)

xgb.plot_importance(bst);

test_df = pd.read_csv('test.csv')

test_df.head()

test_df['shop_id'] = test_df['shop_id'].astype('str')
test_df['item_id'] = test_df['item_id'].astype('str')

test_df = test_df.merge(train_cleaned_df, how = 'left', on=['shop_id', 'item_id']).fillna(0.0)
test_df.head()

d = dict(zip(test_df.columns[4:], list(np.array(list(test_df.columns[4:])) -1)))

test_df  = test_df.rename(columns = d)
test_df.head()

preds = model.predict(X_test)

X_test = test_df.drop(['ID', -1], axis=1).values
print(X_test.shape)

sub_df = pd.DataFrame({'ID':test_df.ID, 'item_cnt_month': preds.clip(0. ,20.)})
sub_df.to_csv('submission.csv',index=False)
