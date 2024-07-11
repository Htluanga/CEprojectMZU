import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/Huma/OneDrive/Documents/GitHub/CEprojectMZU/dataset.csv'
df = pd.read_csv(file_path)


# Rename columns to fit Prophet's requirements
df.rename(columns={'DATE': 'ds','PRICE': 'y'}, inplace=True)

# Ensure the date column is in datetime format
df['ds'] = pd.to_datetime(df['ds'])

# Initialize the Prophet model
model = Prophet()

# Addind holiday for India
model.add_country_holidays(country_name='IN')

# Dataframe copy and exclude the time between 9 PM to 6 AM

df2=df.copy()

df2['ds'] = pd.to_datetime(df2['ds'])

df2=df2[df2['ds'].dt.hour > 6]

df3=df2.copy()

df3['ds'] = pd.to_datetime(df3['ds'])

df3=df3[df3['ds'].dt.hour < 19]


# Fit the model on the dataset
model.fit(df3)

# Assuming df is your DataFrame
df3_description = df3.describe()

# Print the summary statistics
print(df3_description)

# Create a dataframe with future dates for forecasting
future = model.make_future_dataframe(periods=2*365*24, freq='H') 

# Filter out the specific times in the future dataframe
future = future[(future['ds'].dt.hour > 7) & (future['ds'].dt.hour < 19)]

# Make the forecast
forecast = model.predict(future)

# Display the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
fig = model.plot(forecast)
fig2 = model.plot_components(forecast)

# Adjust the x-axis format to include time
fig.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
fig.autofmt_xdate()

plt.show()