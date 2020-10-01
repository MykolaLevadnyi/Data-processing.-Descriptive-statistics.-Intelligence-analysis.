import pandas as pd

df = pd.read_csv('D:\Study\Data-processing.-Descriptive-statistics.-Intelligence-analysis\Birthweight.csv')
pd.set_option('display.max_rows', 140)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width',600)
print(df.head())
df['weight'].describe() #quartiles, min, max, standart deviation
df['weight'].median() # median
df['weight'].var() #variation
df['weight'].std()/df['weight'].mean() #variation coefficient
df['weight'].max()-df['weight'].min() #variation range
df["weight"].quantile(0.75)-df["weight"].quantile(0.25) #quantile range
df["weight"].quantile([0.1,0.9]) #deciles
df['weight'].skew() #coef assymetry
df['weight'].kurtosis() #coef excess
box=df.boxplot(column='weight')

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import hist

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
for i, bins in enumerate(['scott', 'freedman']):
    hist(df["weight"], bins=bins, ax=ax[i], histtype='stepfilled',
         alpha=0.5, density=False)
    ax[i].set_xlabel('weight')
    ax[i].set_ylabel('')
    ax[i].set_title('{0}'.format(bins),
                    fontdict=dict(family='monospace'))

plt.hist(df["weight"], bins='sturges')
plt.xlabel("weight")

import seaborn as sb
from matplotlib import pyplot as plt
sb.distplot(df['weight'])
plt.show()

import numpy as np
import pylab
import scipy.stats as stats

stats.probplot(df['weight'], dist="norm", plot=pylab)
pylab.title("Q-Q plot of weight")
pylab.show()

import statsmodels.api as sm
from matplotlib import pyplot as plt

pp = sm.ProbPlot(df['weight'])
pp.ppplot(line="45")
plt.title("P-P plot of weight")
plt.show()


from scipy.stats import shapiro
data = df['weight']
# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
