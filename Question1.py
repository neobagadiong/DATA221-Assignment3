import pandas

#create dataframe from csv
crimeStatsDF = pandas.read_csv('crime.csv')

#printout
print('Mean:',crimeStatsDF['ViolentCrimesPerPop'].mean())
print('Median:',crimeStatsDF['ViolentCrimesPerPop'].median())
print('SD:',crimeStatsDF['ViolentCrimesPerPop'].std())
print('Min:',crimeStatsDF['ViolentCrimesPerPop'].min())
print('Max:',crimeStatsDF['ViolentCrimesPerPop'].max())

'''
• Compare the mean and median. Does the distribution look symmetric or skewed? Explain
briefly.
    - The distribution of the given data set should be right skewed as the mean is greater than the median making more than half of the data points are under the mean value.

• If there are extreme values (very large or very small), which statistic is more affected: mean
or median? Explain why.
    -  Mean would be more affected by outlier values because it directly uses the values to calculate while median is only determined by the quantity of data points. 
'''