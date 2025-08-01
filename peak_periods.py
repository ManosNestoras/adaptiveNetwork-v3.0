import pandas as pd
from datetime import timedelta

# Read the CSV file
file_path = r"C:\Users\Manos Nestoras\OneDrive-KYTE\adaptiveNetwork\M.Antipa_algo-test\models\mean_week_tcID6.xlsx"
df = pd.read_excel(file_path)

# Calculate mean, std, and threshold for each sensor
stats = {}
sensor_columns = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8']

for h in sensor_columns:
    mean = df[h].mean()
    std = df[h].std()
    threshold = mean + 2 * std
    stats[h] = {'mean': mean, 'std': std, 'threshold': threshold}

# Function to get peak timezones filtered by min duration and max gap
def get_peak_timezones_filtered(df, sensor, threshold, max_gap_minutes=15, min_duration_minutes=30):
    filtered = df[df[sensor] > threshold][['day', 'hour', 'minute']].copy()
    
    if filtered.empty:
        return pd.DataFrame(columns=["Sensor", "Day", "Start Time", "End Time"])

    # Convert hour and minute to time objects for easier calculations
    filtered['time'] = pd.to_datetime(filtered['hour'].astype(str).str.zfill(2) + ':' +
                                      filtered['minute'].astype(str).str.zfill(2), format='%H:%M')

    # Group by day
    grouped = filtered.groupby('day')

    peak_periods = []

    for day, group in grouped:
        group = group.sort_values('time').reset_index(drop=True)
        
        # Initialize first period
        start_time = group.loc[0, 'time']
        end_time = group.loc[0, 'time']
        
        for i in range(1, len(group)):
            current_time = group.loc[i, 'time']
            diff = (current_time - end_time).seconds / 60  
            
            if diff <= max_gap_minutes:
                # Extend the current period
                end_time = current_time
            else:
                # Check the total duration of the period
                duration = (end_time - start_time).seconds / 60
                if duration >= min_duration_minutes:
                    peak_periods.append([sensor, day, start_time.strftime('%H:%M'), end_time.strftime('%H:%M')])
                # Start a new period
                start_time = current_time
                end_time = current_time
        
        # Append the last period if it meets the min duration
        duration = (end_time - start_time).seconds / 60
        if duration >= min_duration_minutes:
            peak_periods.append([sensor, day, start_time.strftime('%H:%M'), end_time.strftime('%H:%M')])

    return pd.DataFrame(peak_periods, columns=["Sensor", "Day", "Start Time", "End Time"])

# Collect peak periods for all sensors
all_peak_periods_filtered = pd.DataFrame()

for h in stats.keys():
    threshold = stats[h]['threshold']
    peaks_df = get_peak_timezones_filtered(df, h, threshold)
    all_peak_periods_filtered = pd.concat([all_peak_periods_filtered, peaks_df], ignore_index=True)

# Sort the results by Sensor, Day (Monday to Sunday), and Start Time
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
all_peak_periods_filtered['Day'] = pd.Categorical(all_peak_periods_filtered['Day'], categories=day_order, ordered=True)

sorted_peak_periods = all_peak_periods_filtered.sort_values(by=['Sensor', 'Day', 'Start Time']).reset_index(drop=True)

# Save sorted results to CSV
output_file_sorted = r"C:\Users\Manos Nestoras\OneDrive-KYTE\adaptiveNetwork\M.Antipa_algo-test\models\peaks_tcID6.xlsx"
sorted_peak_periods.to_excel(output_file_sorted, index=False)

print(f"Sorted peak periods saved to: {output_file_sorted}")