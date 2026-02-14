import matplotlib.pyplot as pyplot
import pandas
crimeStatsDF = pandas.read_csv('crime.csv')

violentCrimesData = crimeStatsDF['ViolentCrimesPerPop']

fig = pyplot.figure(figsize =(10,7))


pyplot.subplot(2,1,1)
pyplot.title('Distribution of Violent Crimes per Population')
pyplot.hist(violentCrimesData, bins=int(len(violentCrimesData)**0.5), color='skyblue', edgecolor='black') #bin size = sqrt of n (data points)
pyplot.ylabel('Frequency')
pyplot.xlabel("Violent Crimes per Population")

pyplot.subplot(2,1,2)
pyplot.boxplot(violentCrimesData,orientation='horizontal')
pyplot.xlabel("Violent Crimes per Population")
pyplot.ylabel("Populations")

# Display the plot
pyplot.show()

'''
• What the histogram shows about how the data values are spread
    - The histogram shows a distribution that is skewed right but with a high number of data points on the top end.

• What the box plot shows about the median
    - Based on the median, the boxplot shows that Q1 and Q2 are more concentrated closer to the
      median value. Q3 and Q4 on the other hand, are more greatly spread out above the median value. 

• Whether the box plot suggests the presence of outliers
    - The box plot's whiskers cover the whole range of data. This means outliers aren't present and 
      the highest end of the data is within the upperlimit based on the middle 50% of the data points.
      Outlier would be plotted as points outside of the whiskers, as the boxplot doesn't 
      have any, there are no outliers.
'''