import numpy as np
from scipy import stats

nobs1=nobs2=10

mean1=1.45907173
std1 = 0.02355322

mean2=1.47341772
std2=0.01968254

modified_std1 = np.sqrt(np.float32(nobs1)/np.float32(nobs1-1)) * std1
modified_std2 = np.sqrt(np.float32(nobs2)/np.float32(nobs2-1)) * std2

(statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=modified_std1, nobs1=10, mean2=mean2, std2=modified_std2, nobs2=10)

print(pvalue)