
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
    /Users/rafael/anaconda3/envs/flatiron-env/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)



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
  <th>Date:</th>             <td>Wed, 27 May 2020</td> <th>  Prob (F-statistic):</th> <td>8.39e-214</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:57:44</td>     <th>  Log-Likelihood:    </th> <td> -13786.</td> 
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


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAH0CAYAAADFQEl4AAAgAElEQVR4Xu3dCZClVXk//gNUgWjYgwRkQINKBCogrkg0IQqIfyOIlgKJQYqIAeIKiEskEFlEVDQoglqyBrRcEWSrxBK3aAwKISoYY9gcFGWZYGSJyL/O+7OnmmFwevo555nb7/3cKitGbn/P25/nMHO+fe97e7UHHnjggeJBgAABAgQIECBAgACBBIHVFJAEZUsQIECAAAECBAgQIDAIKCA2AgECBAgQIECAAAECaQIKSBq1hQgQIECAAAECBAgQUEDsAQIECBAgQIAAAQIE0gQUkDRqCxEgQIAAAQIECBAgoIDYAwQIECBAgAABAgQIpAkoIGnUFiJAgAABAgQIECBAQAGxBwgQIECAAAECBAgQSBNQQNKoLUSAAAECBAgQIECAgAJiDxAgQIAAAQIECBAgkCaggKRRW4gAAQIECBAgQIAAAQXEHiBAgAABAgQIECBAIE1AAUmjthABAgQIECBAgAABAgqIPUCAAAECBAgQIECAQJqAApJGbSECBAgQIECAAAECBBQQe4AAAQIECBAgQIAAgTQBBSSN2kIECBAgQIAAAQIECCgg9gABAgQIECBAgAABAmkCCkgatYUIECBAgAABAgQIEFBA7AECBAgQIECAAAECBNIEFJA0agsRIECAAAECBAgQIKCA2AMECBAgQIAAAQIECKQJKCBp1BYiQIAAAQIECBAgQEABsQcIECBAgAABAgQIEEgTUEDSqC1EgAABAgQIECBAgIACYg8QIECAAAECBAgQIJAmoICkUVuIAAECBAgQIECAAAEFxB4gQIAAAQIECBAgQCBNQAFJo7YQAQIECBAgQIAAAQIKiD1AgAABAgQIECBAgECagAKSRm0hAgQIECBAgAABAgQUEHuAAAECBAgQIECAAIE0AQUkjdpCBAgQIECAAAECBAgoIPYAAQIECBAgQIAAAQJpAgpIGrWFCBAgQIAAAQIECBBQQOwBAgQIECBAgAABAgTSBBSQNGoLESBAgAABAgQIECCggNgDBAgQIECAAAECBAikCSggadQWIkCAAAECBAgQIEBAAbEHCBAgQIAAAQIECBBIE1BA0qgtRIAAAQIECBAgQICAAmIPECBAgAABAgQIECCQJqCApFFbiAABAgQIECBAgAABBcQeIECAAAEC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CkjcsFmCDd2MUhABAgTSBBSQNOpuCykg3WhHGey8Fh+rAhI3bJZgQzejFESAAIE0AQUkjbrbQgpIN9pRBjuvxceqgJRSPvShDw3/uf766wfRbbfdthx11FFljz32GP7/e++9txx++OHl/PPPL3fffXd57nOfW0499dSy+eabL53AjTfeWGhRS88AACAASURBVA499NDyxS9+say99tplv/32K+9+97vLmmuuOecp2dBzpvJEAgQITIyAAjIxo5j3hSgg86abyi90XouPXQEppVx44YVljTXWKI9//OMH0bPOOqucdNJJ5Tvf+c5QRg4++ODhOWeeeWbZaKONymGHHVZuv/32cuWVVw5fd//995cddtihbLzxxuU973lPue2228r+++9f9t5773LKKafMeUo29JypPJEAAQITI6CATMwo5n0hCsi86abyC53X4mNXQB7GcMMNNxxKyEtf+tKhWJxzzjnl5S9/+fDsxYsXl0WLFpWLL7647L777uWSSy4pL3zhC8tNN91UNttss+E5H//4x8srX/nKcuutt5Z11113TpOyoefE5EkECBCYKAEFZKLGMa+LUUDmxTa1X+S8Fh+9ArKMYX0145Of/OTwCkZ9BeQnP/nJ8Jar+orHBhtssPTZ22+/fdlrr73KMcccM7xd64ILLihXX3310n9+xx13lFpi6luydtlllzlNyoaeE5MnESBAYKIEFJCJGse8LkYBmRfb1H6R81p89ArIbwyvueaastNOO5V77rmn/M7v/E4577zzygte8ILh/x5wwAHDfSCzH7vttlt53OMeV04//fRy0EEHDfePXH755Q96zlprrTW8bWvfffdd7qRq5uzcuqHrKytLliyZ86sm8S0ggQABAgQiAgpIRG8yvlYBmYw5LJSrUEDik1JAfmN43333lXoj+Z133lk+/elPl49+9KPliiuuKFddddVyC8iuu+5attpqq3LaaacNBeSGG24ol1122YMmUm9AP/vss8s+++yz3EkdffTRwysoyz4UkPjGlkCAAIEsAQUkS7rfOgpIP9sxJisg8akqIA9j+LznPW8oGPW+j15vwfIKSHwDSyBAgMCqFlBAVvUE4usrIHHDaUpQQOLTVkAexrCWjvp2qPe///3DTejnnntuednLXjY8+5Zbbhk+gnfZm9Bvvvnmsummmw7P+cQnPjHcR+Im9Pgm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" width="800">



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


```python

```
