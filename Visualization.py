# Imports
import matplotlib.pyplot as plt
import pandas as pd

# Data plot
# Create the X axis (Time) for the length that Connor jumped
dfConnor = pd.read_csv(r"ConnorData.csv")
xFirstAxis = dfConnor["Time (s)"]

# Create the X axis (Time) for the length of time Elizabeth jumped
dfElizabeth = pd.read_csv(r"ElizabethData.csv")
xSecondAxis = dfElizabeth["Time (s)"]

# Plot each team member's data with respect to the length of time they collected data for
plt.plot(xFirstAxis, dfConnor['Linear Acceleration z (m/s^2)'], label = "Connor's Data (Pocket)", color = "pink")
plt.plot(xSecondAxis, dfElizabeth["Linear Acceleration z (m/s^2)"], label = "Elizabeth's Data (Hand)")

# Axis labels
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")

# Legend placement
plt.legend(loc = "upper right")
# Title and grid
plt.title("Linear Acceleration Z Data vs Time")
plt.grid(True)
plt.show()

# Meta-Data plot
# Calculate the amount of data in each member's data frames and the total size of the data frame
sizes = [len(dfConnor), len(dfElizabeth)]
total = sum(sizes)

# Labels and title
labels = ["Connor's Data", "Elizabeth's Data"]
plt.title("The Trial's Contribution to Total Collected Data")

# Plot the amount of data each team member contributed
plt.pie(sizes, labels = labels, autopct='%1.1f%%')

# Display the total amount of data collected
plt.text(2, -1.15, f'Whole represents: {total} segments', ha='right', va = 'bottom' )

plt.show()
