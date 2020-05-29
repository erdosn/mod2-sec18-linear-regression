
# Objectives
YWBAT
* list the assumptions of OLS
* explain why multicollinearity is bad both predictively and mathematically
* summarize a statsmodels summary of an OLS


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as scs

import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv("ames.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



# Assumptions of Linear Regression
* linearity: the relationship between the predictors and the target is linear
* Homoskedacicity: the variance of the residuals is equivalent for any given X. Error is normally distributed. 
* X values are independent. All of the rows of data are independent from one another.
    * no multicollinearity
    * use linear regression on your features to check for this (VIF)
* Y is normally distributed


# To do modeling it's best to follow the OSEMN Process
<img src="images/osemn.jpeg"/>

* [X] Obtain
* [ ] Scrub, skip
* [X] Explore
* [X] Model 
* [X] Interpret


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB



```python
# Explore it
numerical_cols = []
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64]:
        numerical_cols.append(col)

len(numerical_cols), numerical_cols[:5]
```




    (38, ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual'])




```python
numerical_df = df[numerical_cols]
numerical_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



# categorical data is data that is a category and order would not make sense


```python
numerical_df.isna().sum()/numerical_df.shape[0]
```




    Id               0.000000
    MSSubClass       0.000000
    LotFrontage      0.177397
    LotArea          0.000000
    OverallQual      0.000000
    OverallCond      0.000000
    YearBuilt        0.000000
    YearRemodAdd     0.000000
    MasVnrArea       0.005479
    BsmtFinSF1       0.000000
    BsmtFinSF2       0.000000
    BsmtUnfSF        0.000000
    TotalBsmtSF      0.000000
    1stFlrSF         0.000000
    2ndFlrSF         0.000000
    LowQualFinSF     0.000000
    GrLivArea        0.000000
    BsmtFullBath     0.000000
    BsmtHalfBath     0.000000
    FullBath         0.000000
    HalfBath         0.000000
    BedroomAbvGr     0.000000
    KitchenAbvGr     0.000000
    TotRmsAbvGrd     0.000000
    Fireplaces       0.000000
    GarageYrBlt      0.055479
    GarageCars       0.000000
    GarageArea       0.000000
    WoodDeckSF       0.000000
    OpenPorchSF      0.000000
    EnclosedPorch    0.000000
    3SsnPorch        0.000000
    ScreenPorch      0.000000
    PoolArea         0.000000
    MiscVal          0.000000
    MoSold           0.000000
    YrSold           0.000000
    SalePrice        0.000000
    dtype: float64




```python

```

# Let's just build an OLS model


```python
cols = ["LotArea", "BsmtFinSF1", "GrLivArea", "MasVnrArea", "Fireplaces"]
```


```python
numerical_df_samp = numerical_df[cols]
numerical_df_samp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>BsmtFinSF1</th>
      <th>GrLivArea</th>
      <th>MasVnrArea</th>
      <th>Fireplaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>8450</td>
      <td>706</td>
      <td>1710</td>
      <td>196.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9600</td>
      <td>978</td>
      <td>1262</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11250</td>
      <td>486</td>
      <td>1786</td>
      <td>162.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>9550</td>
      <td>216</td>
      <td>1717</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>14260</td>
      <td>655</td>
      <td>2198</td>
      <td>350.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# What is the goal of linear regression? 
* can we make a linear equation to predict the saleprice? 

saleprice_hat = B0 + B1xLotArea + B2xBsmtFinSF1 + ... + B5xFireplaces

* adding multiple terms makes it polynomial
* what is making this equation linear? 
    * the features are all a power of 1
* what are we solving for when we do linear regression? 
    * beta coefficients


```python
X = numerical_df[cols]
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>BsmtFinSF1</th>
      <th>GrLivArea</th>
      <th>MasVnrArea</th>
      <th>Fireplaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>8450</td>
      <td>706</td>
      <td>1710</td>
      <td>196.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9600</td>
      <td>978</td>
      <td>1262</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11250</td>
      <td>486</td>
      <td>1786</td>
      <td>162.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>9550</td>
      <td>216</td>
      <td>1717</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>14260</td>
      <td>655</td>
      <td>2198</td>
      <td>350.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = sm.add_constant(X)
X.head()
```

    /Users/rafael/anaconda3/envs/flatiron-env/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>LotArea</th>
      <th>BsmtFinSF1</th>
      <th>GrLivArea</th>
      <th>MasVnrArea</th>
      <th>Fireplaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>8450</td>
      <td>706</td>
      <td>1710</td>
      <td>196.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.0</td>
      <td>9600</td>
      <td>978</td>
      <td>1262</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>11250</td>
      <td>486</td>
      <td>1786</td>
      <td>162.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.0</td>
      <td>9550</td>
      <td>216</td>
      <td>1717</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.0</td>
      <td>14260</td>
      <td>655</td>
      <td>2198</td>
      <td>350.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = numerical_df['SalePrice']
y.head()
```




    0    208500
    1    181500
    2    223500
    3    140000
    4    250000
    Name: SalePrice, dtype: int64




```python
numerical_df.dropna(axis=0, inplace=True)
X = numerical_df[cols] # create your X data
X = sm.add_constant(X) # add a constant
y = numerical_df['SalePrice'] # got our target data
```

    /Users/rafael/anaconda3/envs/flatiron-env/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
ols = sm.OLS(y, X)
results = ols.fit()
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>SalePrice</td>    <th>  R-squared:         </th> <td>   0.591</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.590</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   322.8</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 29 May 2020</td> <th>  Prob (F-statistic):</th> <td>8.39e-214</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:23:45</td>     <th>  Log-Likelihood:    </th> <td> -13786.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1121</td>      <th>  AIC:               </th> <td>2.758e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1115</td>      <th>  BIC:               </th> <td>2.761e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td> 2.364e+04</td> <td> 5165.184</td> <td>    4.576</td> <td> 0.000</td> <td> 1.35e+04</td> <td> 3.38e+04</td>
</tr>
<tr>
  <th>LotArea</th>    <td>    0.5483</td> <td>    0.210</td> <td>    2.611</td> <td> 0.009</td> <td>    0.136</td> <td>    0.960</td>
</tr>
<tr>
  <th>BsmtFinSF1</th> <td>   29.9726</td> <td>    3.676</td> <td>    8.154</td> <td> 0.000</td> <td>   22.760</td> <td>   37.185</td>
</tr>
<tr>
  <th>GrLivArea</th>  <td>   81.7819</td> <td>    3.755</td> <td>   21.777</td> <td> 0.000</td> <td>   74.413</td> <td>   89.151</td>
</tr>
<tr>
  <th>MasVnrArea</th> <td>   84.5858</td> <td>    9.448</td> <td>    8.953</td> <td> 0.000</td> <td>   66.048</td> <td>  103.123</td>
</tr>
<tr>
  <th>Fireplaces</th> <td> 1.451e+04</td> <td> 2915.011</td> <td>    4.977</td> <td> 0.000</td> <td> 8787.470</td> <td> 2.02e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>484.168</td> <th>  Durbin-Watson:     </th> <td>   1.985</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>25222.017</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.203</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>26.113</td>  <th>  Cond. No.          </th> <td>4.27e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.27e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



# Why does multicollinearity ruin a model?
* if 1 feature is correlated with 2 or more features than a change in that feature will cause change in the other features. 
* it creates more weight towards the correlated features 
* there is no signal coming from correlated features 
* there is no solution mathematically to your system of equations

# Visualize Residuals! 


```python
residuals = results.resid
```


```python
# histogram
```


```python
plt.figure(figsize=(8, 5))
plt.hist(residuals)
plt.show()
```


![png](linear-regression_files/linear-regression_24_0.png)



```python
# scatterplot
```


```python
x = np.linspace(0, 1, len(residuals))
```


```python
plt.figure(figsize=(8, 5))
plt.scatter(x, residuals)
plt.xlabel("x axis")
plt.ylabel("residuals")
plt.hlines(xmin=0, xmax=1, y=0, label="y_hat")
plt.show()
```


![png](linear-regression_files/linear-regression_27_0.png)



```python
# We want our scatterplot
# evenly spaced around 0
# no outliers
```


```python
scs.kstest(scs.zscore(residuals), 'norm', args=(0, 1))
```




    KstestResult(statistic=0.08271013001644745, pvalue=4.047882956342313e-07)




```python
scs.shapiro(residuals)
```




    (0.8694653511047363, 1.4159442969666034e-29)



### What did we learn today? 
* in linear regression we're looking for beta coefficients 
* osemn process
    * great process for building models
* the 4 assumptions of linear regression
    * linearity
    * homoskedacicity
    * y is normally distributed
    * xvalues are independent 
* why multicollinearity ruins a model
* continuous and categorical data 

## Objectives
YWBAT (You Will Be Able To)
* transform data to normalize it 
    * explain the rational behind normalizing data
* explain what standardization does
* Test for multicollinearity using a VIF Test

# Transformations
* normalization - make data more like a normal distribution
* standardization - scaling the data 


```python
numerical_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



# Transform data - normalization
* log transormations
* boxcox transformations


```python
for col in numerical_df.columns:
    plt.hist(numerical_df[col], bins=20)
    plt.title(col)
    plt.show()
```


![png](linear-regression_files/linear-regression_35_0.png)



![png](linear-regression_files/linear-regression_35_1.png)



![png](linear-regression_files/linear-regression_35_2.png)



![png](linear-regression_files/linear-regression_35_3.png)



![png](linear-regression_files/linear-regression_35_4.png)



![png](linear-regression_files/linear-regression_35_5.png)



![png](linear-regression_files/linear-regression_35_6.png)



![png](linear-regression_files/linear-regression_35_7.png)



![png](linear-regression_files/linear-regression_35_8.png)



![png](linear-regression_files/linear-regression_35_9.png)



![png](linear-regression_files/linear-regression_35_10.png)



![png](linear-regression_files/linear-regression_35_11.png)



![png](linear-regression_files/linear-regression_35_12.png)



![png](linear-regression_files/linear-regression_35_13.png)



![png](linear-regression_files/linear-regression_35_14.png)



![png](linear-regression_files/linear-regression_35_15.png)



![png](linear-regression_files/linear-regression_35_16.png)



![png](linear-regression_files/linear-regression_35_17.png)



![png](linear-regression_files/linear-regression_35_18.png)



![png](linear-regression_files/linear-regression_35_19.png)



![png](linear-regression_files/linear-regression_35_20.png)



![png](linear-regression_files/linear-regression_35_21.png)



![png](linear-regression_files/linear-regression_35_22.png)



![png](linear-regression_files/linear-regression_35_23.png)



![png](linear-regression_files/linear-regression_35_24.png)



![png](linear-regression_files/linear-regression_35_25.png)



![png](linear-regression_files/linear-regression_35_26.png)



![png](linear-regression_files/linear-regression_35_27.png)



![png](linear-regression_files/linear-regression_35_28.png)



![png](linear-regression_files/linear-regression_35_29.png)



![png](linear-regression_files/linear-regression_35_30.png)



![png](linear-regression_files/linear-regression_35_31.png)



![png](linear-regression_files/linear-regression_35_32.png)



![png](linear-regression_files/linear-regression_35_33.png)



![png](linear-regression_files/linear-regression_35_34.png)



![png](linear-regression_files/linear-regression_35_35.png)



![png](linear-regression_files/linear-regression_35_36.png)



![png](linear-regression_files/linear-regression_35_37.png)



```python
lot_frontage = numerical_df['LotFrontage']
plt.hist(lot_frontage)
plt.show()
```


![png](linear-regression_files/linear-regression_36_0.png)



```python
log_lot_frontage = np.log(lot_frontage)
plt.hist(log_lot_frontage)
plt.show()
```


![png](linear-regression_files/linear-regression_37_0.png)



```python
np.exp(5) # e = 2.718.... 2.718^? = 148.413
```




    148.4131591025766




```python
np.log(148.4131591025766)
```




    5.0




```python
np.log(200)
```




    5.298317366548036




```python
np.log(300), np.log(200)
```




    (5.703782474656201, 5.298317366548036)



# Comparing the skewness and kurtosis we see that both scores get much closer to 0


```python
scs.skew(lot_frontage), scs.kurtosis(lot_frontage)
```




    (2.2481832382080698, 18.356938419863447)




```python
scs.skew(log_lot_frontage), scs.kurtosis(log_lot_frontage)
```




    (-0.7453562715504678, 2.4194490925703844)



# Log Transformations
* Pros
    * normalizes data 
* Cons
    * Interpreting a log is difficult

# Box Cox Transformation


```python
bc_lot_frontage, lam = scs.boxcox(lot_frontage)
```


```python
plt.hist(bc_lot_frontage)
plt.show()
```


![png](linear-regression_files/linear-regression_48_0.png)



```python
scs.skew(bc_lot_frontage), scs.kurtosis(bc_lot_frontage)
```




    (0.12110392077998999, 3.3689474840388636)



# Boxcox in Sklearn - without standardization


```python
from sklearn.preprocessing import PowerTransformer
```


```python
pt = PowerTransformer(method='box-cox', standardize=False)
```


```python
lot_frontage.values
```




    array([65., 80., 68., ..., 66., 68., 75.])




```python
bc_lot_frontage = pt.fit_transform(lot_frontage.values.reshape(-1, 1))
```


```python
plt.hist(bc_lot_frontage)
plt.show()
```


![png](linear-regression_files/linear-regression_55_0.png)


# Boxcox in Sklearn - with standardization


```python
pt = PowerTransformer(method='box-cox', standardize=True)
```


```python
bc_lot_frontage = pt.fit_transform(lot_frontage.values.reshape(-1, 1))
plt.hist(bc_lot_frontage)
plt.show()
```


![png](linear-regression_files/linear-regression_58_0.png)


# Workflow w/o train test split


```python
X = numerical_df.drop(columns='SalePrice')
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>548</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>460</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>608</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>642</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>836</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
y = numerical_df['SalePrice']
```


```python
X['log_lot_frontage'] = np.log(X['LotFrontage'])
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>log_lot_frontage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>4.174387</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>4.382027</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>4.219508</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>4.094345</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>4.430817</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
X['log_garage_area'] = np.log(X['GarageArea'])
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>log_lot_frontage</th>
      <th>log_garage_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>4.174387</td>
      <td>6.306275</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>4.382027</td>
      <td>6.131226</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>4.219508</td>
      <td>6.410175</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>4.094345</td>
      <td>6.464588</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>4.430817</td>
      <td>6.728629</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
X['bc_gc_living_area'], lam = scs.boxcox(X['GrLivArea'])
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>log_lot_frontage</th>
      <th>log_garage_area</th>
      <th>bc_gc_living_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>4.174387</td>
      <td>6.306275</td>
      <td>5.812015</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>4.382027</td>
      <td>6.131226</td>
      <td>5.628985</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>4.219508</td>
      <td>6.410175</td>
      <td>5.837900</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>4.094345</td>
      <td>6.464588</td>
      <td>5.814450</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>4.430817</td>
      <td>6.728629</td>
      <td>5.960381</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
X['SalePrice'] = y
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>log_lot_frontage</th>
      <th>log_garage_area</th>
      <th>bc_gc_living_area</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>4.174387</td>
      <td>6.306275</td>
      <td>5.812015</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>4.382027</td>
      <td>6.131226</td>
      <td>5.628985</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>4.219508</td>
      <td>6.410175</td>
      <td>5.837900</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>4.094345</td>
      <td>6.464588</td>
      <td>5.814450</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>4.430817</td>
      <td>6.728629</td>
      <td>5.960381</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



#### Build a model with the columns untransformed and compare it to a model with transformed columns


```python
ut_cols = ['LotFrontage', 'GarageArea', 'GrLivArea']
t_cols = ['log_lot_frontage', 'log_garage_area', 'bc_gc_living_area']
```


```python
def build_ols(df=numerical_df, 
              cols=['LotFrontage', 'GarageArea', 'GrLivArea'], 
              target='SalePrice',
              add_constant=False):
    x = df[cols]
    if add_constant:
        x = sm.add_constant(x)
    y = df[target]
    ols = sm.OLS(y, x)
    res = ols.fit()
    print(res.summary())
    return res
```


```python
def plot_residuals(resids):
    plt.hist(resids)
    plt.title("Residuals")
    plt.show()
    
    xspace = np.linspace(0, 1, len(resids))
    plt.scatter(x, resids)
    plt.title("resids")
    plt.hlines(0, xmin=0, xmax=1)
    plt.show()
```


```python
build_ols(df=numerical_df, cols=ut_cols[1:])
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:              SalePrice   R-squared (uncentered):                   0.932
    Model:                            OLS   Adj. R-squared (uncentered):              0.932
    Method:                 Least Squares   F-statistic:                              7645.
    Date:                Fri, 29 May 2020   Prob (F-statistic):                        0.00
    Time:                        12:23:53   Log-Likelihood:                         -13786.
    No. Observations:                1121   AIC:                                  2.758e+04
    Df Residuals:                    1119   BIC:                                  2.759e+04
    Df Model:                           2                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    GarageArea   141.1718      8.680     16.264      0.000     124.141     158.202
    GrLivArea     76.0093      2.886     26.340      0.000      70.347      81.671
    ==============================================================================
    Omnibus:                      243.863   Durbin-Watson:                   2.037
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5600.299
    Skew:                           0.385   Prob(JB):                         0.00
    Kurtosis:                      13.923   Cond. No.                         9.73
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





    <statsmodels.regression.linear_model.OLS at 0x1c1b360208>



# Interpret the OLS Summary
* Section A - model
    * R2 -> Explained Variance 
    * P(F) < 0.05
    * AIC, BIC - regularization scores
* Section B - features
    * pvalues
        * H0: the coefficient predicts the target randomly
            * this happens with
                * bad predictors
                * multicollinearity
                * non normal data
        * HA: the coefficient predicts the target not randomly
* Section C - residuals
    * Are my residuals normal?
        * skew -> 0 is ideal
        * kurtosis -> 0 is ideal 
    * Cond. No -> multicollinearity smaller is better


```python
build_ols(df=X, cols=t_cols[1:])
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:              SalePrice   R-squared (uncentered):                   0.861
    Model:                            OLS   Adj. R-squared (uncentered):              0.861
    Method:                 Least Squares   F-statistic:                              3461.
    Date:                Fri, 29 May 2020   Prob (F-statistic):                        0.00
    Time:                        12:23:53   Log-Likelihood:                         -14186.
    No. Observations:                1121   AIC:                                  2.838e+04
    Df Residuals:                    1119   BIC:                                  2.839e+04
    Df Model:                           2                                                  
    Covariance Type:            nonrobust                                                  
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    log_garage_area    6.101e+04   6523.394      9.353      0.000    4.82e+04    7.38e+04
    bc_gc_living_area -3.287e+04   7030.184     -4.675      0.000   -4.67e+04   -1.91e+04
    ==============================================================================
    Omnibus:                      525.029   Durbin-Watson:                   2.017
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3753.860
    Skew:                           2.032   Prob(JB):                         0.00
    Kurtosis:                      10.990   Cond. No.                         35.5
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





    <statsmodels.regression.linear_model.OLS at 0x1c1b3786a0>



# Day 3
Sklearn, cross validation, feature engineering, recommendations and future work

YWBAT 
* build a model in statsmodels but verify and validate it in sklearn
* develop a workflow in sklearn
* feature engineering (next week)
* remove Outliers programmatically
* perform Recursive feature selection


```python
final_df = X.copy()
final_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>log_lot_frontage</th>
      <th>log_garage_area</th>
      <th>bc_gc_living_area</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>4.174387</td>
      <td>6.306275</td>
      <td>5.812015</td>
      <td>208500</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>4.382027</td>
      <td>6.131226</td>
      <td>5.628985</td>
      <td>181500</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>4.219508</td>
      <td>6.410175</td>
      <td>5.837900</td>
      <td>223500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>4.094345</td>
      <td>6.464588</td>
      <td>5.814450</td>
      <td>140000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>4.430817</td>
      <td>6.728629</td>
      <td>5.960381</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>




```python
final_df.to_csv("final_df.csv", index=False)
```


```python
t_cols, ut_cols
```




    (['log_lot_frontage', 'log_garage_area', 'bc_gc_living_area'],
     ['LotFrontage', 'GarageArea', 'GrLivArea'])




```python
build_ols(final_df, cols=ut_cols[1:], target='SalePrice')
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:              SalePrice   R-squared (uncentered):                   0.932
    Model:                            OLS   Adj. R-squared (uncentered):              0.932
    Method:                 Least Squares   F-statistic:                              7645.
    Date:                Fri, 29 May 2020   Prob (F-statistic):                        0.00
    Time:                        13:12:09   Log-Likelihood:                         -13786.
    No. Observations:                1121   AIC:                                  2.758e+04
    Df Residuals:                    1119   BIC:                                  2.759e+04
    Df Model:                           2                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    GarageArea   141.1718      8.680     16.264      0.000     124.141     158.202
    GrLivArea     76.0093      2.886     26.340      0.000      70.347      81.671
    ==============================================================================
    Omnibus:                      243.863   Durbin-Watson:                   2.037
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5600.299
    Skew:                           0.385   Prob(JB):                         0.00
    Kurtosis:                      13.923   Cond. No.                         9.73
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





    <statsmodels.regression.linear_model.OLS at 0x1c1c8bd048>




```python
model = build_ols(final_df, cols=ut_cols[1:], target='SalePrice')
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:              SalePrice   R-squared (uncentered):                   0.932
    Model:                            OLS   Adj. R-squared (uncentered):              0.932
    Method:                 Least Squares   F-statistic:                              7645.
    Date:                Fri, 29 May 2020   Prob (F-statistic):                        0.00
    Time:                        13:14:26   Log-Likelihood:                         -13786.
    No. Observations:                1121   AIC:                                  2.758e+04
    Df Residuals:                    1119   BIC:                                  2.759e+04
    Df Model:                           2                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    GarageArea   141.1718      8.680     16.264      0.000     124.141     158.202
    GrLivArea     76.0093      2.886     26.340      0.000      70.347      81.671
    ==============================================================================
    Omnibus:                      243.863   Durbin-Watson:                   2.037
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5600.299
    Skew:                           0.385   Prob(JB):                         0.00
    Kurtosis:                      13.923   Cond. No.                         9.73
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
resids = model.resid
```


```python
plot_residuals(resids)
```


![png](linear-regression_files/linear-regression_81_0.png)



![png](linear-regression_files/linear-regression_81_1.png)


# Build Model in Sklearn


```python
X = final_df[ut_cols[1:]]
y = final_df['SalePrice']
```


```python
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from sklearn.metrics import r2_score
```


```python
# What is a train test split? 
# train/test ratios: 70/30 - 90/10 
# rare cases(60/40, 95/5)

# train/test = 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
```


```python
# build the model 

# instantiate your model
# fit_intercept: fit a B0 or don't fit a B0
# normalize: Normalizes the beta coefficients (almost always set to False)
# copy_X: leave it alone
# n_jobs: -1 -> parallelize the work on all your computer cores
ols = LinearRegression(fit_intercept=False, normalize=False, n_jobs=-1)
```


```python
ols.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=-1, normalize=False)




```python
ols.coef_
```




    array([146.01791821,  74.71788749])




```python
ols.intercept_
```




    0.0




```python
# let's get our training predictions
y_train_preds = ols.predict(X_train)
y_train_preds
```




    array([175263.00041953, 213260.98092683, 280557.87255581, 197656.05280739,
           144088.55205338, 174559.30375994, 173036.17827209, 148841.25415316,
           197856.18834503, 142415.51582923, 212634.75879521, 201748.35420792,
           202550.69375235, 250021.23164807, 168266.38688849, 256298.29786474,
           103261.7316468 , 176473.20109652, 146446.295808  , 187644.23771791,
           123577.68636857, 149287.95033916, 235051.73618042, 115206.43320199,
           333853.93774146, 226235.31416327, 172139.56362224, 163083.12796387,
           146661.71391162, 304863.97480903, 193786.48018535, 131723.3609887 ,
           122456.91805626, 183909.38639365, 293626.09906381, 243733.15751737,
            93828.19556676, 232122.45600242, 222034.11630761, 145528.2240949 ,
           132259.58266684, 236694.86848887, 274651.74158752, 116702.69079706,
           227380.28685535, 164200.29216448, 241443.10968177, 189607.97802197,
           197506.52390494, 174512.97177669, 240559.87775259, 225531.23566982,
           130677.49681882, 313641.66422418, 327993.00257882, 260375.21424784,
           137633.95286388, 150119.71883796, 179756.42036656, 234368.83536779,
           319067.39540015, 159750.06572635, 134146.61211064, 205314.78062806,
           165410.11100761, 181583.92136137, 170883.50592333, 173109.94623692,
           293585.46661649, 175358.79353641, 123875.60799587, 257372.82732128,
           173483.06071303, 228330.35790835, 348606.31083909, 111510.03509733,
            83169.93058749, 180089.56361162, 305760.77571383, 228751.22925169,
           123291.53632302, 118347.43424444, 204416.2661329 , 275323.54135864,
           297773.27242751, 176185.53303952, 199764.67485207, 173937.54253522,
           246260.27609874, 159270.90456308, 117204.73390756, 172326.21398777,
           119020.84515448, 133364.77970679, 167342.04755067, 190703.77828681,
           215337.32424917, 192373.48978167, 138509.49228434, 167358.09378436,
           226168.00007811, 158210.23278854, 152320.71614275, 276766.81751187,
           202098.49360933, 227899.13986725, 157945.0648449 , 171426.56331501,
           189586.52095873, 235584.35374685, 293871.23482819, 167828.33313385,
           246689.97612839, 148978.91834639, 150142.60078519, 192837.36837896,
           169245.59818948, 162576.62394648, 161757.67007955, 124469.55140515,
           125251.29068152, 143401.94467761, 388025.16983768, 110404.83805738,
           132148.31140508, 177106.07268672, 248053.41227097, 254219.10477444,
           186716.86235718, 135107.78420516, 179911.1738436 , 285018.35256422,
           180730.220838  , 179475.58802308, 179333.93788429, 301699.14189671,
           219484.49761287, 337176.65328121, 218741.30468355, 139262.74320504,
           172868.61016317, 168552.15510018,  87938.20395964, 174972.95755589,
           105943.31499884, 111867.01021227, 143893.64109034, 183439.14704416,
           155527.69951384, 209411.63606295, 229006.23675251, 294499.35680511,
           199748.5354909 , 207854.71384131, 159865.70476754, 241969.64587842,
           323414.70430149, 210678.42229348, 230414.10503302, 230747.72323942,
           103866.78542155, 210489.49024878, 210933.42979429, 192290.4181692 ,
           175328.03282553, 112564.91420847, 211592.508061  , 168552.15510018,
           160786.33754221, 161468.57713857, 112295.28535794, 320070.53169849,
           118448.54506338, 183591.90754843, 140524.02547854, 112295.28535794,
           151816.29822561, 195060.20458084, 182375.43924671, 146888.81046952,
           157587.52164116, 198149.64906544, 170795.59157012, 192105.28581512,
           213604.19148724, 298815.8118681 , 182170.84280218, 306919.04732308,
           166775.15828912, 249362.8239216 , 108297.64089669, 134539.28380468,
           207664.83187397, 152119.53755498, 111771.31022287, 169001.41234776,
           246900.16965741, 124688.29423806, 367801.02228722, 117206.15879154,
           192182.56476419, 280780.69446176, 284541.65933506, 173099.12457783,
           134149.55500607, 257399.69521407,  85254.72076228, 248540.54532538,
           190228.6030691 , 131755.73283851, 170134.33407568, 157124.59296652,
           201652.65421852, 193774.04738733, 153610.47744797, 278104.42881179,
           288086.15373064, 268061.75988396, 110359.93095812, 153847.35261483,
           302082.98490444, 100170.48044436, 355538.83256283, 174967.07176502,
           224905.581627  , 186057.50470805, 176798.46557793, 209523.00045219,
           335837.90580369, 242921.42119782, 190703.77828681, 114995.38287779,
           167196.41146631, 166237.80043337, 162723.20995349, 259710.06393531,
           166976.24374941, 201946.11493892, 155184.96391475, 167916.81557587,
           163117.78149284, 144121.30573704, 132729.91514381, 120588.02094641,
           155925.78203743, 262744.34775035, 160928.08080848, 173981.5928393 ,
           117122.70534521, 239882.00593565, 118092.5198711 , 122748.47893135,
           229439.44776639, 164741.54283831, 181387.39925941, 279913.61121772,
           265065.44500321, 221334.31246611, 110195.87383343, 359792.15506513,
           219354.43280088, 136795.44177902, 216219.31754931, 112295.28535794,
           192940.47217069, 189399.8705932 , 241731.8207889 , 164069.08185092,
           212730.07695076, 233322.40310911, 247875.30188537, 190265.81765965,
           129880.38184899, 322267.72931266, 230148.17342168, 146949.76380248,
           227465.64014698, 142851.10164975, 245951.1509785 , 207410.39246195,
           212926.79463163, 233876.09590493, 172503.94253951, 168757.22650604,
           206454.15623565, 151694.5871386 , 121709.73918138, 196296.89131676,
            99518.52659756, 137856.11355356, 280594.80776394, 228066.70797617,
           154219.33091334, 149556.15430571,  84378.61325302, 229806.2014903 ,
           240045.11313769, 158595.1188464 , 136649.51698828, 197617.88829418,
           232277.20947947, 200023.00708217, 247116.44455622, 139034.12863568,
           277596.40678295, 166181.50358622, 115859.81193278, 239697.91663171,
           163008.41007638, 169748.11626131, 168967.80186892, 120065.47069532,
           152283.97651352, 172220.1673006 , 122868.6720069 , 247365.56626616,
           168934.00513513, 177681.79063458, 117246.69811138, 133215.43705929,
           243457.64287597, 205619.53796889, 193051.07289222, 273340.04825773,
           198607.35316547, 129825.22117943, 184096.89355437, 197820.58489341,
           121490.99634846, 161462.21638637, 151352.41962832, 191650.14277667,
           163387.69904975, 132505.85460882, 175066.75769999, 248632.25936921,
           628611.72922811, 189133.27776559, 170075.75554936, 163024.93127141,
           206045.91326924, 210882.54377668, 203064.98340592, 231896.69120103,
           141108.57211272, 147399.68226633, 443969.31034797, 199330.98887685,
           160800.57705807, 115831.99411734, 251381.24993389, 167984.60462236,
           109224.54129609, 205837.89896795, 286300.89632206, 171092.56327476,
           170665.71301307, 137501.1314114 , 165800.60347392, 156150.88562255,
           199657.38953586, 214172.59876024,  99518.52659756, 125204.95869828,
           186115.60827304, 169689.15590113, 234283.57520363, 152949.12682605,
           154600.23102564, 246195.90490902, 136315.04198671, 116610.79049827,
           138087.57789088, 158029.94317521, 282970.39514612, 148078.50400592,
           177140.72621569, 141546.71899482, 159398.31518601, 208179.02840007,
           182172.26768616, 196590.25890969, 268273.76013082, 211702.54069373,
           190714.98177976, 184177.40410525, 132162.55092093, 141300.44705284,
           195187.41962486, 209417.99681515, 198646.37447386,  73897.60186417,
           154805.01372511, 162425.2883262 , 143387.79828922, 268170.3676327 ,
           268525.34977487, 211636.37211014, 114909.46149735, 134907.9373739 ,
           191985.75395584, 183924.86453855, 215012.05976776, 192821.79710659,
           111134.92764844, 293491.28463855, 183053.31106365, 151686.23341362,
           131045.67542671, 124025.9936935 , 260455.34296488, 162424.33840354,
           177653.11602395, 207785.40678337, 185516.91525048, 199374.08925828,
           147782.48222393, 190445.4460567 , 299875.05875865, 159750.06572635,
           130982.0679047 , 147633.04644896, 133753.9404166 , 322219.21810169,
           256385.73725663, 239012.73413992, 151598.50531535, 116661.10842708,
           132984.82951715, 146386.67423155, 107139.7511213 , 181992.07120031,
           224818.61719644, 157621.31837495, 243405.90006317, 242523.04996785,
           324203.46554633, 180990.45291342, 220150.02975925, 205239.58777925,
           143521.09470303, 227125.84744333, 186363.305099  , 103252.52112663,
           124915.29766849, 167145.5254487 , 148662.57567876, 172306.08868104,
           282938.49825765, 152575.53738862, 239997.07688804, 190703.77828681,
           246014.75850051, 209198.21093211, 168591.17640857, 213362.85541348,
           185658.09042795, 220277.25412724, 179768.57378216, 233512.85316526,
           273325.24065308, 166865.82928283, 152016.43376325, 155254.27097269,
           388838.23791373, 183179.67863646, 168059.98372612, 114122.21826396,
           311731.37099931, 190355.15689685, 226566.93939688, 272470.11524573,
           182629.9717862 , 206261.8063342 , 178974.01987391, 217505.56787029,
           196018.52690739, 178826.95890557, 198967.74613718, 327036.29139119,
           183179.67863646, 148469.56456104, 478460.68159047, 307939.74786657,
           265499.60593975, 114897.21495428, 112302.5960328 , 157445.01470719,
           133513.26555911, 195734.84479596, 233439.56016175, 199974.20716482,
           176059.54730056,  95117.48191065, 217272.58552152, 152398.47005314,
           193366.27938225, 109924.3451376 , 338152.26047049, 179236.33804958,
           236663.72594414, 148609.78981584, 134770.748142  , 213306.26985994,
           176399.43313169, 167170.01853485, 203942.51579917, 147887.3927335 ,
           244465.52878759, 159075.99360004, 168911.1231879 , 121268.74253131,
           145920.89578895, 146094.1634338 , 133777.48358009, 183900.17587349,
           225465.72830249, 165942.34674019, 158914.97249826, 136918.48462254,
           244343.62212167, 216845.2602985 , 211425.12620702, 109299.25918358,
           197040.65233487, 298136.51516718, 164818.63553244, 121818.63563651,
           201911.46140995, 309184.60207253, 230868.77311015, 150829.3944159 ,
           122015.92140619, 150016.04695743, 158021.77570517, 169897.07707496,
           219603.55451082, 165053.42459905, 280054.40456133, 172139.56362224,
           217944.18971372, 188191.37418262, 249071.16991903, 154579.81701252,
           183871.69684178, 211388.9546666 , 108701.51608368, 189998.08865443,
           140950.4007789 , 189586.52095873, 156773.1218086 , 141898.19015275,
           289028.24356855, 255683.55860849, 142976.51929991, 234749.35364623,
           173037.12819474, 181322.07814703, 204854.22676006, 286728.1284176 ,
           176773.02256913, 116895.04069851, 199097.1497329 , 226521.65046376,
           201428.02559473, 167763.96194413, 102768.51722261, 138827.4460909 ,
           225932.6429227 , 201893.51533094, 282346.25911476, 160738.96250884,
           186411.63005503, 183179.67863646, 230199.823107  , 167819.12261369,
           204417.69101688, 168552.15510018, 158992.06519238, 192070.15732483,
           156287.59989312, 119467.72759542, 228310.8938179 , 170719.92375997,
           149674.261281  , 119030.71689092, 231877.41336551, 149637.70790672,
           200830.28249482, 200360.04314533, 159127.2614515 , 144198.87339249,
           177967.55884627, 135809.01293064, 249035.18463355, 107670.08700857,
           166237.80043337, 107356.5941089 , 148498.51855408, 180120.41744997,
           152692.40573488, 139918.97170379, 218246.86095429, 260874.88255171,
           135599.95557922, 149750.59030742, 129555.11736758, 278379.28223692,
           284106.64160333, 129555.11736758, 116356.44421373, 136113.77027147,
           105441.27188834, 265361.75549158, 167325.3401007 , 174100.36103087,
           154020.23842583, 165253.74639163, 111717.5744373 , 109841.84161393,
           143071.74432798, 141014.2970073 , 205243.57372481, 300952.43798316,
           240310.5697877 , 286427.7388562 , 311976.50676368, 151247.79782513,
           156297.85346341, 248543.96318214, 153145.64892802, 122963.42207365,
           216499.58180398, 155387.18555265, 156641.1571513 , 179155.16628242,
           119189.36318606, 222806.73819129, 105767.01133108, 168393.79751142,
           228536.94732567, 114318.55411098, 190579.31055931, 115889.0546322 ,
           114834.74360987, 208077.44261979, 195307.04461162, 249732.04557962,
           216248.56024873, 216219.31754931, 165736.2322842 , 142011.07255344,
           188614.71346006, 139331.38904671, 218609.81498758, 268003.56319149,
           252873.99654471, 154042.26357787,  93759.83843147, 194924.15152654,
           214569.07014491, 392237.18935243, 173037.12819474, 111145.2743462 ,
           115703.06548295, 173204.12821487, 179801.80242715, 212648.9051836 ,
           261514.11489411, 106062.93998559, 183617.63926362, 144983.64869178,
           201502.74348221, 245063.74684881, 212624.41209745, 111391.36003323,
           184954.96185715, 143896.20215192, 181119.38154781, 246297.4906893 ,
           148006.6358864 , 186189.46936535, 110195.87383343, 201577.93633103,
           191308.35710024, 282881.34461531,  91661.37682961, 148379.27540118,
           220817.0799171 , 210890.80437419, 160930.93057644, 267729.65968902,
           193407.86175222, 114461.15417243, 158463.62915043, 120364.8172066 ,
           163174.36704638, 298220.72295725, 188273.21649002, 181701.9352092 ,
           189018.02055825, 201707.52618169,  88155.9968699 , 130749.65364473,
           169246.92994599, 122680.12179606, 219439.49738613, 199215.25670819,
           192521.88250652, 276521.58862002, 178781.19501112, 151367.99090068,
           199280.28911419,  99955.06234074, 224564.8390007 , 162874.54557378,
           317085.61588961, 236354.98265775, 276673.49232911, 188113.62027222,
           285254.84589723, 108361.63025256, 149582.54723717, 207427.86357963,
           194852.85149582, 179977.91983996, 182856.97521663, 106609.41523403,
           198648.84240797,  75959.8919256 , 182900.83926576, 181863.14256592,
           197930.43127119,  92801.22739853, 158041.71475696, 249504.56718786,
           257145.73076338, 189564.11397283, 155995.5640567 , 250953.92471088,
           298916.44772572, 239116.7878543 , 232051.1559717 , 126042.90169433,
           305356.33243804, 304948.65756043, 298634.85171453, 178983.41664902,
           115799.715395  , 154408.54234045, 171321.46655049, 217509.55381585,
           162427.18817151, 194681.86553013, 177307.15814701, 241811.85637846,
           316622.68721497, 198825.90974344, 205466.20937582, 119167.43116149,
           112215.15664091, 177816.22322599, 157726.32201199, 207971.10722625,
           177433.33014091, 115206.43320199, 215159.1207361 , 168656.59064842,
           168581.3977996 , 182261.13196204, 258222.06693776, 212214.74424705,
           453149.83189998, 161812.92387658, 227134.01491337, 130370.65337774,
           189419.90277246, 135111.58389578, 129730.94607401, 171430.93109443,
           196403.88792658, 206309.6563289 , 147150.56055639, 204719.9804236 ,
           246089.00142667, 211560.32246614, 200885.91812571, 208922.22132938,
           246483.86167241, 135455.93063379, 160381.80113895, 141653.71560464,
           151651.10492332, 180654.93486172, 108351.85164359, 240809.94938518,
           258400.74541216, 169367.69111034, 162365.19178842, 202337.83671031,
           172587.87094716, 168852.92649544, 144677.9414283 , 227032.81096695,
           171854.36349935, 339184.45321332,  68646.84259945, 289721.21169652,
           194430.93710235, 212558.14106241, 193808.7009163 , 157506.9179628 ,
           188696.74202241, 327475.67690233, 182192.77482675, 167196.41146631,
           178937.94146096, 150434.72974908, 180981.71735458, 111771.31022287,
           179105.89140374, 266439.04158862, 156756.98244743, 127505.45568308,
           179498.08813645, 128444.69575303, 215984.81718908, 172094.74965045,
           151026.96889196, 183104.96074897, 235571.2504086 , 160674.02323032,
           115590.18308225, 256326.68376898, 170405.48093766, 239249.13434546,
           280387.92964025, 185365.5796302 , 171426.56331501, 174226.34676982,
           134880.11955846, 212313.29400442, 248661.50206864, 144059.30935396,
           120140.66354414, 163591.53182657, 219020.43276063, 229994.18361234,
           106325.16503379, 147915.39680389, 188816.84197049, 208370.9033402 ,
           188503.25594335, 273168.01924192, 132505.85460882, 166460.05425053,
           159750.06572635, 127718.49898007, 212092.17636487, 170644.25594983,
           240984.64191402, 118804.57025567, 161379.23790137, 233999.32500339,
           106134.71497764, 235526.25018186, 159035.07244634, 196819.91652917,
           120310.13149837, 164965.69650077, 113709.51439066, 187572.46272586,
           248976.1311459 , 364913.0357728 , 169626.02334045, 213737.48790105,
           172179.62798076, 200430.77508725, 198915.91019692, 234180.37828443,
           197638.39543477, 171916.35988243, 204169.23052322, 227382.56853451,
           209744.11809175, 205389.02355422, 318839.73075345, 300566.22016879,
           188475.91308924, 280023.64385045, 218138.43946049, 193814.58670717,
           189714.49967046, 228291.81156129, 263049.48692503, 202042.85797845])




```python
r2_score(y_train, y_train_preds)
```




    0.5853904131179467




```python
def get_score(model ,X_train, y_train):
    y_train_preds = model.predict(X_train)
```


```python
# r2 
training_score = ols.score(X_train, y_train)
# ols.score is calculating the y_train_preds and then using those to get the r2 score
training_score
```




    0.5853904131179467




```python
# is this model overfitting or underfitting or neither?
```


```python
testing_score = ols.score(X_test, y_test)
testing_score
```




    0.6069516655669215




```python
# traininig_score < testing_score by a pretty significant amount
# this means our model is basically random

# ideally your scores are really close together
```


```python
# train test splits are really good indicators of model quality
# but if your scores fluctuate between runs this means the data is ill conditioned or 
# you got a bad train/test split
```


```python
# to avoid this, do cross validation
```


```python
# cv = 5 means 80/20 splits
# cv = 10 means 90/10 splits
# cv = 3 means 67/33 splits

cv_scores = cross_val_score(ols, X, y, scoring='r2', cv=5, n_jobs=-1) 
# cv breaks your data into that many equal pieces 
cv_scores
```




    array([0.71352907, 0.6114661 , 0.62620083, 0.54991384, 0.40749908])




```python
# shuffle the data then cross fold
X_shuff, y_shuff = shuffle(X, y)
```


```python
cv_scores = cross_val_score(ols, X_shuff, y_shuff, scoring='r2', cv=5, n_jobs=-1) 
# cv breaks your data into that many equal pieces 
cv_scores
```




    array([0.55089841, 0.62435084, 0.64351805, 0.42044351, 0.64509527])




```python
# Scaling our data
ss = StandardScaler()
```


```python
ss.fit(X_train)
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```python
X_train_scaled = ss.transform(X_train)
```


```python
X_test_scaled = ss.transform(X_test)
```


```python
ols.fit(X_train_scaled, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=-1, normalize=False)




```python
train_scaled_score = ols.score(X_train_scaled, y_train)
train_scaled_score
```




    -4.335263227674017




```python
test_scaled_score = ols.score(X_test_scaled, y_test)
test_scaled_score
```




    -5.731559112985546




```python
def make_ols_sklearn(X, y, test_size=0.20, fit_intercept=False, standardize=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    if standardize:
        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)
    ols = LinearRegression(fit_intercept=fit_intercept, normalize=False)
    ols.fit(X_train, y_train)
    train_score = ols.score(X_train, y_train)
    test_score = ols.score(X_test, y_test)
    print(f"train score = {train_score}")
    print(f"test score = {test_score}")
    return ols
```


```python
make_ols_sklearn(X, y)
```

    train score = 0.5657280722618787
    test score = 0.6732944774952542





    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)




```python
make_ols_sklearn(X, y, standardize=True)
```

    train score = -4.240205635517639
    test score = -5.499647895526162





    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)




```python
X2 = final_df[t_cols[1:]]
```


```python
make_ols_sklearn(X2, y)
```

    train score = 0.16509945735316522
    test score = 0.16310934741932093





    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)




```python
make_ols_sklearn(X2, y, standardize=True)
```

    train score = -4.334162027974564
    test score = -4.686673813110008





    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)




```python
# Feature Forward Selection
final_df['GrLivArea'].plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c21acc3c8>




![png](linear-regression_files/linear-regression_115_1.png)



```python
final_df = final_df.loc[final_df['GrLivArea']<2500]
```


```python
X = final_df[ut_cols[1:]]
y = final_df['SalePrice']
X2 = final_df[t_cols[1:]]
```


```python
make_ols_sklearn(X, y)
```

    train score = 0.599978562461299
    test score = 0.5308538526390951





    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)




```python
make_ols_sklearn(X2, y)
```

    train score = 0.19031001900641464
    test score = 0.19325759105538354





    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)




```python
final_df['GrLivArea'].plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c22032588>




![png](linear-regression_files/linear-regression_120_1.png)


# using RFE to find the best 10 features


```python
rfe = RFE(ols, n_features_to_select=20, step=1, verbose=2)
```


```python
X_all = final_df.drop(columns=['SalePrice'])
rfe.fit(X_all, y)
```

    Fitting estimator with 40 features.
    Fitting estimator with 39 features.
    Fitting estimator with 38 features.
    Fitting estimator with 37 features.
    Fitting estimator with 36 features.
    Fitting estimator with 35 features.
    Fitting estimator with 34 features.
    Fitting estimator with 33 features.
    Fitting estimator with 32 features.
    Fitting estimator with 31 features.
    Fitting estimator with 30 features.
    Fitting estimator with 29 features.
    Fitting estimator with 28 features.
    Fitting estimator with 27 features.
    Fitting estimator with 26 features.
    Fitting estimator with 25 features.
    Fitting estimator with 24 features.
    Fitting estimator with 23 features.
    Fitting estimator with 22 features.
    Fitting estimator with 21 features.





    RFE(estimator=LinearRegression(copy_X=True, fit_intercept=False, n_jobs=-1,
                                   normalize=False),
        n_features_to_select=20, step=1, verbose=2)




```python
rfe.support_
```




    array([False,  True, False, False,  True,  True,  True, False, False,
           False, False, False, False, False, False, False,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True, False, False, False, False,  True, False, False, False,
           False,  True,  True,  True])




```python
new_cols = X_all.columns[rfe.support_]
```


```python
X_new = final_df[new_cols]
```


```python
make_ols_sklearn(X_new, y)
```

    train score = 0.8509010477662157
    test score = 0.8718407568368854





    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)




```python
pd.plotting.scatter_matrix(X_new, figsize=(20, 20))
plt.show()
```


![png](linear-regression_files/linear-regression_128_0.png)



```python

```
