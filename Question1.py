import pandas
crimeStatsDF = pandas.read_csv('crime.csv')

print('Mean:',crimeStatsDF['ViolentCrimesPerPop'].mean())
print('Median:',crimeStatsDF['ViolentCrimesPerPop'].median())
print('SD:',crimeStatsDF['ViolentCrimesPerPop'].std())
print('Min:',crimeStatsDF['ViolentCrimesPerPop'].min())
print('Max:',crimeStatsDF['ViolentCrimesPerPop'].max())