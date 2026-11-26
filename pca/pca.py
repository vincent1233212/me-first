#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


# In[142]:


data = pd.read_csv('log_returns_v2.csv')
data = data.set_index('date').sort_index()
returns = data.drop(columns=['^GSPC','^VIX']) #drop the marco-indicators, keep the 50 stock returns only
returns = returns.loc['1994-07-08':]
stock_names = returns.columns


# In[144]:


returns.head()


# In[145]:


x_train = returns.loc[:'2019-12-31'] #train test spilt
x_test = returns.loc['2020-01-01':]


# In[146]:


scaler = StandardScaler()
z_train = scaler.fit_transform(x_train)
z_test = scaler.transform(x_test)

z_train = pd.DataFrame(z_train)
z_test = pd.DataFrame(z_test)


# In[147]:


corr = x_train.corr()
p = corr.shape[0]
average_corr = (corr.to_numpy().sum()-p)/(p*(p-1))

sns.heatmap(corr,cmap='RdBu_r',center=0,vmin=-1,vmax=1)
plt.title('Correlation Matrix of 50 Selected Stocks')
plt.tight_layout()
plt.show()


# In[148]:


pca = PCA()
pca.fit(z_train)
explained_var = pca.explained_variance_ratio_
cumu_var = np.cumsum(explained_var)
eigenvalue = pca.explained_variance_

plt.bar(range(1,11),explained_var[:10]*100)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance(%)')
plt.show()


# In[149]:


plt.plot(range(1,11),cumu_var[:10]*100,marker='o')
plt.ylim(0,100)
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance')
plt.show()


# In[197]:


pc_no = np.arange(1,len(eigenvalue)+1)
plt.plot(pc_no,eigenvalue,marker = "o")
plt.axhline(y = 1, color = 'orange',label = 'Eigenvalue=1')
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.legend()
plt.show()


# In[152]:


kaiser_count = np.sum(eigenvalue > 1)
print('Number of PCs under Kaiser criterion:',kaiser_count)


# In[153]:


loadings = pd.DataFrame(pca.components_.T, index = stock_names, columns=[f"PC{i}" for i in range(1,51)])


# In[199]:


sector_groups = {'Communication Services': ['CMCSA', 'DIS', 'OMC', 'T'],
                 'Consumer Discretionary': ['HAS', 'LOW', 'MCD', 'NKE', 'ROST'],
                 'Consumer Staples': ['ADM', 'CL', 'KMB', 'PEP'],
                 'Energy': ['APA', 'COP', 'CVX', 'VLO', 'XOM'],
                 'Financials': ['AXP', 'SPGI', 'TFC', 'TRV'],
                 'Health Care': ['BMY', 'CI', 'DHR', 'MDT', 'UNH'],
                 'Industrials': ['HON', 'MMM', 'NDSN', 'PAYX', 'SWK'],
                 'Information Technology': ['HPQ', 'ORCL', 'TXN', 'WDC'],
                 'Materials': ['APD', 'NUE', 'SHW', 'VMC'],
                 'Real Estate': ['DOC', 'ESS', 'KIM', 'WY'],
                 'Utilities': ['DTE', 'DUK', 'ED', 'EXC', 'NEE', 'PCG']}

sector_map = {stock: sector 
              for sector, stocks in sector_groups.items() 
              for stock in stocks}

selected_loading = loadings.iloc[:, :7]
squared_loading = selected_loading**2

sector_contribution = squared_loading.groupby(sector_map).sum() * 100


# In[199]:


sector_groups = {'Communication Services': ['CMCSA', 'DIS', 'OMC', 'T'],
                 'Consumer Discretionary': ['HAS', 'LOW', 'MCD', 'NKE', 'ROST'],
                 'Consumer Staples': ['ADM', 'CL', 'KMB', 'PEP'],
                 'Energy': ['APA', 'COP', 'CVX', 'VLO', 'XOM'],
                 'Financials': ['AXP', 'SPGI', 'TFC', 'TRV'],
                 'Health Care': ['BMY', 'CI', 'DHR', 'MDT', 'UNH'],
                 'Industrials': ['HON', 'MMM', 'NDSN', 'PAYX', 'SWK'],
                 'Information Technology': ['HPQ', 'ORCL', 'TXN', 'WDC'],
                 'Materials': ['APD', 'NUE', 'SHW', 'VMC'],
                 'Real Estate': ['DOC', 'ESS', 'KIM', 'WY'],
                 'Utilities': ['DTE', 'DUK', 'ED', 'EXC', 'NEE', 'PCG']}

sector_map = {stock: sector 
              for sector, stocks in sector_groups.items() 
              for stock in stocks}

selected_loading = loadings.iloc[:, :7]
squared_loading = selected_loading**2

sector_contribution = squared_loading.groupby(sector_map).sum() * 100


# In[200]:


sns.heatmap(sector_contribution,cmap='RdBu_r')
plt.title('Sector Contributions to Principal Components (%)')
plt.xlabel('Principal Component')
plt.ylabel('Sector')
plt.tight_layout()
plt.show()


# In[201]:


sns.heatmap(selected_loading,cmap='RdBu_r')
plt.title('Principal Component Loadings')
plt.xlabel('Principal Component')
plt.ylabel('Stock')
plt.tight_layout()
plt.show()


# In[208]:


pc_train = pca.transform(z_train)
pc_train = pc_train[:, :7]
pc_train = pd.DataFrame(pc_train, columns = [f"PC{i}" for i in range(1,8)],index = x_train.index)

pc_test = pca.transform(z_test)
pc_test = pc_test[:, :7]
pc_test = pd.DataFrame(pc_test, columns = [f"PC{i}" for i in range(1,8)],index = x_test.index)


# In[202]:


for pc in selected_loading.columns:
    plt.figure(figsize=(14, 4))
    loadings[pc].plot(kind='bar')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Loading')
    plt.title(f'Stock Loadings of {pc}')
    plt.tight_layout()
    plt.show()


# In[187]:


pc_all = pd.concat([pc_train, pc_test]).sort_index()

features = pc_all.copy()
features['PC1_vol_5'] = pc_all['PC1'].rolling(5).std()
features['PC1_vol_20'] = pc_all['PC1'].rolling(20).std()

features['PC1_sum_5'] = pc_all['PC1'].rolling(5).sum()
features['PC1_sum_20'] = pc_all['PC1'].rolling(20).sum()


# In[189]:


features['T2'] = (pc_all.pow(2)/ eigenvalue[:7]).sum(axis=1)
features['T2_mean_5'] = features['T2'].rolling(5).mean()


# In[209]:


z_all = pd.concat([z_train, z_test]).to_numpy()
H = pca.components_[:7].T
z_rec = pc_all.to_numpy() @ H.T + pca.mean_
features['SPE'] = np.sum((z_all - z_rec)**2, axis=1)
features['SPE_mean_5'] = features['SPE'].rolling(5).mean()


# In[177]:


factor_corr = pd.concat([pc_train,data[['^GSPC', '^VIX']]], axis=1).loc[pc_train.index].corr()
sns.heatmap(factor_corr,annot=True,fmt='.2f',cmap='RdBu_r',center=0)
plt.title('Correlation between PCs and Market Indicators')
plt.show()


# In[219]:


for feature in features:

    plt.figure(figsize=(14, 4))

    features[feature].loc['2007':'2009'].plot()
    plt.title(feature + ' from 2007 to 2009')
    plt.ylabel(feature)
    plt.show()


# In[233]:


analysis = features.copy()

analysis['future_loss'] = -data['^GSPC'].shift(-1)

analysis['target_date'] = pd.Series(
    data.index, index=data.index
).shift(-1)

train_analysis = analysis.loc[:'2019-12-31'].copy()

train_analysis = train_analysis[
    train_analysis['target_date'] <= '2019-12-31'
].dropna()

predictive_corr = (
    train_analysis.drop(columns='target_date')
    .corr(method='spearman')['future_loss']
    .drop('future_loss')
)

print(predictive_corr.sort_values())


# In[234]:


threshold = train_analysis['future_loss'].quantile(0.99)

train_analysis['extreme_loss'] = (
    train_analysis['future_loss'] > threshold
).astype(int)


# In[239]:


indicators = [
    'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7',
    'PC1_vol_5', 'PC1_vol_20',
    'PC1_sum_5', 'PC1_sum_20',
    'T2', 'T2_mean_5',
    'SPE', 'SPE_mean_5'
]
fig, axes = plt.subplots(
    3, 5,
    figsize=(20, 12)
)

axes = axes.flatten()

for i, indicator in enumerate(indicators):

    sns.boxplot(
        data=train_analysis,
        x='extreme_loss',
        y=indicator,
        showfliers=False,
        ax=axes[i]
    )

    axes[i].set_title(indicator)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['Normal', 'Extreme'])

plt.suptitle(
    'Distribution of PCA Indicators Before Next-day Extreme Losses',
    fontsize=18
)

plt.tight_layout()
plt.show()


# In[203]:


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

results = []

y = train_analysis['extreme_loss']
baseline = y.mean()

for indicator in features.columns:

    x = train_analysis[indicator]

    # Determine predictive direction
    raw_auc = roc_auc_score(y, x)

    direction = 1 if raw_auc >= 0.5 else -1

    score = direction * x

    auc = roc_auc_score(y, score)
    ap = average_precision_score(y, score)

    # Riskiest 20% of observations
    high_risk = score >= score.quantile(0.8)

    conditional_prob = y[high_risk].mean()

    lift = conditional_prob / baseline

    results.append({
        'Indicator': indicator,
        'Risk_Direction': direction,
        'ROC_AUC': auc,
        'Average_Precision': ap,
        'AP_Lift': ap / baseline,
        'Conditional_Probability': conditional_prob,
        'Top_Group_Lift': lift
    })

indicator_results = pd.DataFrame(results)

indicator_results = indicator_results.sort_values(
    'Average_Precision',
    ascending=False
)

display(indicator_results.round(4))


# In[204]:


plt.figure(figsize=(10, 7))

sns.barplot(
    data=indicator_results,
    x='Average_Precision',
    y='Indicator',
    color='steelblue'
)

plt.axvline(
    baseline,
    color='red',
    linestyle='--',
    label='Random Baseline'
)

plt.xlabel('Average Precision')
plt.ylabel('Indicator')
plt.title('Predictive Performance of PCA Indicators')
plt.legend()
plt.tight_layout()
plt.show()


# In[207]:


selected_indicators = [
    'PC1_vol_5',
    'PC1_vol_20',
    'PC1_sum_20',
    'T2_mean_5',
    'SPE'
]

selected_features = features[
    selected_indicators
].dropna()

selected_features.to_csv(
    'pca_selected_indicators.csv',
    index_label='date'
)

