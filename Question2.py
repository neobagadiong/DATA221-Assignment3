import matplotlib.pyplot as pyplot
import pandas
crimeStatsDF = pandas.read_csv('crime.csv')

violentCrimesData = crimeStatsDF['ViolentCrimesPerPop']

fig = pyplot.figure(figsize =(10,7))


pyplot.subplot(2,1,1)
pyplot.title('Distribution of Violent Crimes per Population')
pyplot.hist(violentCrimesData, bins=18, color='skyblue', edgecolor='black')
pyplot.ylabel('Frequency')
pyplot.xlabel("Violent Crimes per Population")

pyplot.subplot(2,1,2)
pyplot.boxplot(violentCrimesData,orientation='horizontal')
pyplot.xlabel("Violent Crimes per Population")
pyplot.ylabel("Populations")

# Display the plot
pyplot.show()