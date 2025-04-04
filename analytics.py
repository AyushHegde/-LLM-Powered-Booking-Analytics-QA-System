import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_hotel_bookings.csv")

# 1. Revenue Trends Over Time
df["arrival_date"] = pd.to_datetime(df["reservation_status_date"])
df["total_revenue"] = df["stays_in_week_nights"] * df["adr"]
monthly_revenue = df.groupby(df["arrival_date"].dt.to_period("M"))["total_revenue"].sum()

plt.figure(figsize=(10, 5))
monthly_revenue.plot(kind="line", marker="o")
plt.title("Revenue Trends Over Time")
plt.ylabel("Total Revenue")
plt.xlabel("Month")
plt.grid()
plt.show()

# 2. Cancellation Rate
cancellation_rate = df["is_canceled"].mean() * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# 3. Geographical Distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=df["country"], order=df["country"].value_counts().index[:10])
plt.title("Top 10 Countries by Bookings")
plt.show()

# 4. Booking Lead Time Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["lead_time"], bins=50, kde=True)
plt.title("Booking Lead Time Distribution")
plt.show()
