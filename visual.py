import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("paddydataset.csv")

# Line chart
df['Paddy yield(in Kg)'].value_counts().plot()
plt.show()

# Histogram
df['Paddy yield(in Kg)'].plot(kind='hist')
plt.show()

# Box plot
df.boxplot(column='Paddy yield(in Kg)')
plt.show()