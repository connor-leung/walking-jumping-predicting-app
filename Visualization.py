import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dfFirst = pd.read_csv(r"ConnorData.csv")
dfSecond = pd.read_csv(r"ElizabethData.csv")

xFirstAxis = dfFirst["Time (s)"]
xSecondAxis = dfSecond["Time (s)"]

plt.plot(xSecondAxis, dfSecond["Acceleration z (m/s^2)"], label = "Elizabeth's Data")
plt.plot(xFirstAxis, dfFirst['Linear Acceleration z (m/s^2)'], label = "Connor's Data", color = "pink")
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
plt.legend(loc = "upper right")
plt.title("Linear Acceleration Z Data vs Time")
plt.grid(True)
plt.show()


sizes = [len(dfFirst), len(dfSecond)]
total = sum(sizes)
labels = ['First Trial', 'Second Trial']
plt.pie(sizes, labels = labels, autopct='%1.1f%%')
plt.title("The Trial's Contribution to Total Collected Data")
plt.text(2, -1.15, f'Whole represents: {total} segments', ha='right', va = 'bottom' )
plt.show()
