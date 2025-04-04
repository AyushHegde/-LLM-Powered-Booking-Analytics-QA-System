import pandas as pd

# Load dataset
df = pd.read_csv("hotel_bookings.csv")

# Display basic info
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop unnecessary columns
df = df.drop(["agent", "company"], axis=1)  # Drop columns with too many missing values

# Fill missing values
df.loc[:, "children"] = df["children"].fillna(0)

# Convert date columns
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce", dayfirst=True)
df["arrival_date"] = df["reservation_status_date"]  # Ensure arrival_date is properly assigned


# Save cleaned dataset
df.to_csv("cleaned_hotel_bookings.csv", index=False)
