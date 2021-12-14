#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import norm
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


train = pd.read_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\train.csv')
print('The shape of training data:', train.shape)
train


# In[41]:


train


# In[42]:


train.columns  # 查看各个特征的具体名称


# In[43]:


test = pd.read_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\test.csv')
print('The shape of testing data:', test.shape)
test.head()


# In[44]:


#绘制目标值分布
sns.distplot(train['SalePrice'])


# In[ ]:





# In[45]:


#分离数字特征和类别特征
num_features = []
cate_features = []
for col in test.columns:
    if test[col].dtype == 'object':
        cate_features.append(col)
    else:
        num_features.append(col)
print('number of numeric features:', len(num_features))
print('number of categorical features:', len(cate_features))


# In[46]:


#查看数字特征与目标值的关系
plt.figure(figsize=(16, 20))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for i, feature in enumerate(num_features):
    plt.subplot(8, 5, i+1)
    sns.scatterplot(x=feature, y='SalePrice', data=train, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
plt.show()


# In[10]:


var,output='OverallQual','SalePrice'
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x=var,y=output,data=train)
ax.set_ylim(0,800000)
plt.show()


# In[47]:


#查看‘Neighborhood’与目标值的关系
plt.figure(figsize=(16, 12))
sns.boxplot(x='Neighborhood', y='SalePrice', data=train)
plt.show()


# In[48]:


corrmat = train.corr()
f,ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corrmat, vmax=.8, square=True, ax=ax)  # square参数保证corrmat为非方阵时，图形整体输出仍为正方形
plt.show()


# In[49]:


cols_10 = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
corrmat_10 = train[cols_10].corr()
plt.figure(figsize=(6, 6))
sns.heatmap(corrmat_10, annot=True)


# In[50]:


g = sns.PairGrid(train[cols_10])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)


# In[51]:


sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)


# In[52]:


train = train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<700000)].index)

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)


# In[53]:


sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)


# In[54]:


train = train.drop(train[(train['GrLivArea']>4000)].index)

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)


# In[55]:


#ID列没有用，直接删掉
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
print('The shape of training data:', train.shape)
print('The shape of testing data:', test.shape)


# In[56]:


#查看训练集中各特征的数据缺失个数
print('The shape of training data:', train.shape)
train_missing = train.isnull().sum()
train_missing = train_missing.drop(train_missing[train_missing==0].index).sort_values(ascending=False)
train_missing


# In[57]:


#查看测试集中各特征的数据缺失个数
print('The shape of testing data:', test.shape)
test_missing = test.isnull().sum()
test_missing = test_missing.drop(test_missing[test_missing==0].index).sort_values(ascending=False)
test_missing


# In[58]:


none_lists = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1',
              'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
for col in none_lists:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')


# In[59]:


most_lists = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual', 'Electrical']
for col in most_lists:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])    #注意这里补充的是训练集中出现最多的类别


# In[60]:


train['Functional'] = train['Functional'].fillna('Typ')
test['Functional'] = test['Functional'].fillna('Typ')

train.drop('Utilities', axis=1, inplace=True)
test.drop('Utilities', axis=1, inplace=True)


# In[61]:


zero_lists = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageCars', 'GarageArea',
              'TotalBsmtSF']
for col in zero_lists:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)


# In[63]:


train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
for ind in test['LotFrontage'][test['LotFrontage'].isnull().values==True].index:
    x = test['Neighborhood'].iloc[ind]
    test['LotFrontage'].iloc[ind] = train.groupby('Neighborhood')['LotFrontage'].median()[x]


# In[64]:


train.isnull().sum().any()


# In[73]:


test.isnull().sum().any()


# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


train['IsRemod'] = 1 
train['IsRemod'].loc[train['YearBuilt']==train['YearRemodAdd']] = 0  #是否翻新(翻新：1， 未翻新：0)
train['BltRemodDiff'] = train['YearRemodAdd'] - train['YearBuilt']  #翻新与建造的时间差（年）
train['BsmtUnfRatio'] = 0
train['BsmtUnfRatio'].loc[train['TotalBsmtSF']!=0] = train['BsmtUnfSF'] / train['TotalBsmtSF']  #Basement未完成占总面积的比例
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']  #总面积
#对测试集做同样的处理
test['IsRemod'] = 1 
test['IsRemod'].loc[test['YearBuilt']==test['YearRemodAdd']] = 0  #是否翻新(翻新：1， 未翻新：0)
test['BltRemodDiff'] = test['YearRemodAdd'] - test['YearBuilt']  #翻新与建造的时间差（年）
test['BsmtUnfRatio'] = 0
test['BsmtUnfRatio'].loc[test['TotalBsmtSF']!=0] = test['BsmtUnfSF'] / test['TotalBsmtSF']  #Basement未完成占总面积的比例
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']  #总面积


# In[76]:


dummy_features = list(set(cate_features).difference(set(le_features)))
dummy_features


# In[80]:


all_data = pd.concat((train.drop('SalePrice', axis=1), test)).reset_index(drop=True)
all_data = pd.get_dummies(all_data, drop_first=True)  


# In[81]:


trainset = all_data[:1456]
y = train['SalePrice']
trainset['SalePrice'] = y.values
testset = all_data[1456:]
print('The shape of training data:', trainset.shape)
print('The shape of testing data:', testset.shape)


# In[82]:


trainset.to_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\train_data.csv', index=False)
testset.to_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\test_data.csv', index=False)


# In[120]:


#基础
import numpy as np
import pandas as pd
import time

#绘图
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#模型
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

#模型相关
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

#忽略警告
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


# In[121]:


train = pd.read_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\train_data.csv')
test = pd.read_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\test_data.csv')


# In[122]:


#查看目标值的斜度和峰度
from scipy.stats import skew, kurtosis, norm

y = train['SalePrice']
print('Skewness of target:', y.skew())
print('kurtosis of target:', y.kurtosis())
sns.distplot(y, fit=norm);


# In[123]:


y = np.log1p(y)
print('Skewness of target:', y.skew())
print('kurtosis of target:', y.kurtosis())
sns.distplot(y, fit=norm);


# In[144]:


#采用十折交叉验证
n_folds = 10

def rmse_cv(model):
  kf = KFold(n_folds, shuffle=True, random_state=20)
  rmse = np.sqrt(-cross_val_score(model, train.values, y, scoring='neg_mean_squared_error', cv=kf))
  return(rmse)


# In[151]:


#Lasso
lasso_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=lasso_alpha, random_state=2))

#ElasticNet
enet_beta = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]
enet_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
ENet = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=enet_beta, alphas=enet_alpha, random_state=12))

#Ridge
rid_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
rid = make_pipeline(RobustScaler(), RidgeCV(alphas=rid_alpha))

#Gradient Boosting
gbr_params = {'loss': 'huber','criterion': 'mse', 'learning_rate': 0.1,
      'n_estimators': 600, 'max_depth': 4,'subsample': 0.6,'min_samples_split': 20,
      'min_samples_leaf': 5,'max_features': 0.6,'random_state': 32,'alpha': 0.5}
gbr = GradientBoostingRegressor(**gbr_params)

#LightGBM
lgbr_params = {'learning_rate': 0.01,'n_estimators': 1850, 'max_depth': 4,
      'num_leaves': 20,'subsample': 0.6,'colsample_bytree': 0.6,'min_child_weight': 0.001,
      'min_child_samples': 21,'random_state': 42,'reg_alpha': 0,'reg_lambda': 0.05}
lgbr = lgb.LGBMRegressor(**lgbr_params)

#XGBoost
xgbr_params = {'learning_rate': 0.01,'n_estimators': 3000, 'max_depth': 5,'subsample': 0.6,
      'colsample_bytree': 0.7,'min_child_weight': 3,'seed': 52,
      'gamma': 0,'reg_alpha': 0,'reg_lambda': 1}
xgbr = xgb.XGBRegressor(**xgbr_params)


# In[152]:


models_name = ['Lasso', 'ElasticNet', 'Ridge', 'Gradient Boosting', 'LightGBM', 'XGBoost']
models = [lasso, ENet, rid, gbr, lgbr, xgbr]
for i, model in enumerate(models):
  score = rmse_cv(model)
  print('{} score: {}({})'.format(models_name[i], score.mean(), score.std()))


# In[155]:


stack_model = StackingCVRegressor(regressors=(lasso, ENet, rid, gbr, lgbr, xgbr), meta_regressor=lasso, use_features_in_secondary=True)


# In[156]:


#Lasso
lasso_trained = lasso.fit(np.array(train), np.array(y))

#ElasticNet
ENet_trained = ENet.fit(np.array(train), np.array(y))

#Ridge
rid_trained = rid.fit(np.array(train), np.array(y))

#Gradient Boosting
gbr_trained = gbr.fit(np.array(train), np.array(y))

#LightGBM
lgbr_trained = lgbr.fit(np.array(train), np.array(y))

#XGBoost
xgbr_trained = xgbr.fit(np.array(train), np.array(y))

#Stacking
stack_model_trained = stack_model.fit(np.array(train), np.array(y))


# In[157]:


def rmse(y, y_preds):
  return np.sqrt(mean_squared_error(y, y_preds))


# In[158]:


models.append(stack_model)
models_name.append('Stacking_model')
for i, model in enumerate(models):
  y_preds = model.predict(np.array(train))
  model_score = rmse(y, y_preds)
  print('RMSE of {}: {}'.format(models_name[i], model_score))


# In[159]:


sample_submission = pd.read_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\sample_submission.csv')
for i, model in enumerate(models):
  preds = model.predict(np.array(test))
  submission = pd.DataFrame({'Id': sample_submission['Id'], 'SalePrice': np.expm1(preds)})
  submission.to_csv(r'D:\我的\课件\大三上期\机器学习\房价预测数据\submission_'+models_name[i]+'_optimation.csv', index=False)
  print('{} finished.'.format(models_name[i]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[145]:





# In[ ]:





# In[ ]:





# In[ ]:




