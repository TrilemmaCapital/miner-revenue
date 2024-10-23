# miner_revenue_animation.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter
from io import StringIO

def plot_daily_rewards_forecast(df, window=365, forecast_years=25, seasonal_periods=365):
    """
    Creates an animated plot of miner rewards, incrementally appending forecast data.
    The animation is saved as a GIF in the 'miner_revenue_animations' folder with a timestamp.

    Parameters:
    df (pd.DataFrame): DataFrame with columns 'FeeTotNtv', 'BlkCnt', and a datetime index.
    window (int): Window size for the moving average of fees.
    forecast_years (int): Number of years to forecast for BlockRewards halving events.
    seasonal_periods (int): Seasonal period for Holt-Winters model.

    Returns:
    None
    """
    print("Starting plot_daily_rewards_forecast function...")

    # Create the folder if it doesn't exist
    output_folder = 'miner_revenue_animations'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder already exists: {output_folder}")

    # Constants for Bitcoin halving and block generation
    BLOCKS_PER_DAY = 144
    HALVING_BLOCK_INTERVAL = 210000
    BLOCK_REWARD_INITIAL = 50  # Initial block reward in BTC

    # Step 1: Obtain the current block height from the BlockCypher API
    print("Step 1: Obtaining the current block height from the BlockCypher API...")
    api_url = "https://api.blockcypher.com/v1/btc/main"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        current_block_height = data['height']
        print(f"Current block height obtained: {current_block_height}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        # Fallback to cumulative sum of BlkCnt
        current_block_height = df['BlkCnt'].cumsum().iloc[-1]
        print(f"Using cumulative sum of 'BlkCnt' as current block height: {current_block_height}")

    # Calculate current halving
    current_halving = current_block_height // HALVING_BLOCK_INTERVAL
    current_block_reward = BLOCK_REWARD_INITIAL / (2 ** current_halving)
    print(f"Current halving: {current_halving}, Current block reward: {current_block_reward} BTC")

    # Step 2: Generate future dates and forecasted block rewards
    print("Step 2: Generating future dates and forecasted block rewards...")
    last_date = df.index[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=forecast_years * 365,
        freq='D'
    )
    print(f"Future dates generated from {future_dates[0]} to {future_dates[-1]}")

    # Calculate future block heights
    days_since_last_block = (future_dates - last_date).days
    future_blocks = days_since_last_block * BLOCKS_PER_DAY + current_block_height
    future_halvings = future_blocks // HALVING_BLOCK_INTERVAL
    future_block_rewards = BLOCK_REWARD_INITIAL / (2 ** future_halvings)
    print("Future block rewards calculated.")

    # Step 3: Create a DataFrame for the forecasted block rewards
    print("Step 3: Creating DataFrame for forecasted block rewards...")
    future_df = pd.DataFrame(index=future_dates)
    future_df['BlockReward'] = future_block_rewards
    future_df['BlkCnt'] = BLOCKS_PER_DAY  # Assume constant block count daily
    print("Future DataFrame created.")

    # Extend the actual data with forecasted block rewards
    df_extended = pd.concat([df, future_df])
    print("Actual data extended with forecasted block rewards.")

    # Step 4: Calculate block rewards in BTC
    print("Step 4: Calculating block rewards in BTC...")
    df_extended['BlockRewards'] = df_extended['BlkCnt'] * df_extended['BlockReward']
    print("Block rewards calculated.")

    # Step 5: Calculate the moving average for Fees
    print("Step 5: Calculating moving average for Fees...")
    df['FeeMovingAvg'] = df['FeeTotNtv'].rolling(window=window, min_periods=1).mean()
    print("Moving average calculated.")

    # Ensure no negative or zero values to avoid log issues
    df['FeeMovingAvg'] = df['FeeMovingAvg'].replace(0, 1e-6)
    # Log-transform for stability
    df['LogFeeMovingAvg'] = np.log(df['FeeMovingAvg'])
    print("Log transformation applied to moving average.")

    # Fit the Holt-Winters model with a damped trend on the moving average of fees
    print("Fitting Holt-Winters model to the moving average of fees...")
    hw_model = ExponentialSmoothing(
        df['LogFeeMovingAvg'].dropna(),
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods,
        damped_trend=True
    )
    hw_fit = hw_model.fit()
    print("Model fitting complete.")

    # Forecast future moving average fees
    print("Forecasting future moving average fees...")
    forecast_steps = len(future_dates)
    moving_avg_forecast = hw_fit.forecast(steps=forecast_steps)
    print("Forecasting complete.")

    # Convert log-forecast back to original scale and fill future moving average fees
    df_extended.loc[future_dates, 'FeeMovingAvg'] = np.exp(moving_avg_forecast)
    # Include historical FeeMovingAvg in df_extended
    df_extended.loc[df.index, 'FeeMovingAvg'] = df['FeeMovingAvg']
    print("Historical and forecasted FeeMovingAvg combined in df_extended.")

    # Step 6: Calculate the total rewards for each day (Fee + Block Rewards)
    print("Step 6: Calculating total rewards for each day...")
    df_extended['TotalRewards'] = df_extended['FeeMovingAvg'].fillna(0) + df_extended['BlockRewards']
    print("Total rewards calculated.")

    # Prepare the data for plotting
    print("Preparing data for plotting...")
    stacked_data = pd.DataFrame({
        'Date': df_extended.index,
        'Fees': df_extended['FeeMovingAvg'].fillna(0),
        'BlockRewards': df_extended['BlockRewards']
    })
    print("Data preparation complete.")

    # Split the data into actual and forecasted
    last_actual_date = last_date
    forecast_data = stacked_data.loc[last_actual_date + pd.Timedelta(days=1):]
    print("Data split into actual and forecasted.")

    # Initialize the plot
    print("Initializing the plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    palette = sns.color_palette()
    print("Plot initialized.")

    # Define y-axis formatter
    def y_axis_formatter(y, pos):
        if y >= 1000:
            return '{:,.0f}'.format(y)
        elif y >= 1:
            return '{:,.0f}'.format(y)
        else:
            return '{:,.1f}'.format(y)

    # Number of frames in the animation
    total_frames = len(forecast_data)
    print(f"Total frames available: {total_frames}")

    # Desired total animation duration in seconds (including pauses)
    total_desired_duration_sec = 15
    pause_duration_start_sec = 2  # Duration of pause at the beginning in seconds
    pause_duration_end_sec = 2    # Duration of pause at the end in seconds
    total_pause_duration_sec = pause_duration_start_sec + pause_duration_end_sec
    desired_duration_sec = total_desired_duration_sec - total_pause_duration_sec
    fps = 30  # Frames per second

    # Calculate the number of frames to match the desired duration
    desired_num_frames = int(desired_duration_sec * fps)
    print(f"Desired number of frames (excluding pauses): {desired_num_frames}")

    # Calculate the number of frames for the pauses
    number_of_pause_frames_start = int(pause_duration_start_sec * fps)
    number_of_pause_frames_end = int(pause_duration_end_sec * fps)
    print(f"Number of pause frames at start: {number_of_pause_frames_start}")
    print(f"Number of pause frames at end: {number_of_pause_frames_end}")

    # If total_frames > desired_num_frames, sample frames accordingly
    if total_frames > desired_num_frames:
        frame_indices = np.linspace(0, total_frames - 1, desired_num_frames).astype(int)
    else:
        frame_indices = np.arange(total_frames)

    # Create the pause frames at the start by repeating the first frame
    pause_frames_start = np.full(number_of_pause_frames_start, frame_indices[0])

    # Create the pause frames at the end by repeating the last frame
    pause_frames_end = np.full(number_of_pause_frames_end, frame_indices[-1])

    # Combine them
    frame_indices = np.concatenate([
        pause_frames_start,
        frame_indices,
        pause_frames_end
    ])
    print(f"Total frames after adding pauses: {len(frame_indices)}")

    # Adjust interval to match the desired duration
    interval = 1000 / fps  # Interval in milliseconds
    print(f"Interval between frames: {interval} ms")

    # Function to update the plot for each frame
    def update(frame_idx):
        frame = frame_indices[frame_idx]
        if frame_idx % 10 == 0:
            print(f"Rendering frame {frame_idx + 1} of {len(frame_indices)}")
        ax.clear()
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        # Set y-axis to logarithmic scale
        ax.set_yscale('log')

        # Set y-ticks and labels
        y_ticks = [0.1, 1, 10, 100, 1000, 10000]
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        # Data up to the current frame
        current_forecast_data = forecast_data.iloc[:frame + 1]
        combined_data = pd.concat([stacked_data.loc[:last_actual_date], current_forecast_data])

        # Plot historical data
        actual_data = stacked_data.loc[:last_actual_date]
        ax.fill_between(
            actual_data['Date'],
            0,
            actual_data['Fees'],
            label='Daily Fees',
            color=palette[0],
            alpha=0.7
        )
        ax.fill_between(
            actual_data['Date'],
            actual_data['Fees'],
            actual_data['Fees'] + actual_data['BlockRewards'],
            label='Daily Block Rewards',
            color=palette[7],
            alpha=0.7
        )

        # Plot forecast data up to current frame
        ax.fill_between(
            current_forecast_data['Date'],
            0,
            current_forecast_data['Fees'],
            color=palette[0],
            alpha=0.3,
            linestyle='--'
        )
        ax.fill_between(
            current_forecast_data['Date'],
            current_forecast_data['Fees'],
            current_forecast_data['Fees'] + current_forecast_data['BlockRewards'],
            color=palette[7],
            alpha=0.3,
            linestyle='--'
        )

        # Plot moving average lines
        ax.plot(
            actual_data['Date'],
            actual_data['Fees'],
            label=f'{window}-day Moving Avg of Fees',
            color='white',
            linewidth=2
        )
        ax.plot(
            current_forecast_data['Date'],
            current_forecast_data['Fees'],
            color='white',
            linewidth=2,
            linestyle='--'
        )

        # Formatting
        ax.set_title('Miner Revenue = Fees + Block Rewards', fontsize=24, color='white')
        ax.set_xlabel('')
        ax.set_ylabel('Rewards in BTC (Log Scale)', fontsize=16, color='white')
        ax.tick_params(axis='x', colors='white', labelsize=14)
        ax.tick_params(axis='y', colors='white', labelsize=14)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=16, loc='lower right', facecolor='black', edgecolor='white', labelcolor='white')

    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, repeat=False)
    print("Animation created.")

    # Save the animation as a GIF with timestamp
    print("Saving animation as GIF...")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')  # e.g., '2023-10-22_13-45'
    gif_filename = f'miner_rewards_forecast_{timestamp}.gif'
    gif_output_path = os.path.join(output_folder, gif_filename)

    # Save as GIF
    anim.save(gif_output_path, writer=PillowWriter(fps=fps))
    print(f'GIF animation saved to {gif_output_path}')

def get_data():
    print("Fetching data from GitHub repository...")
    # Fetch the CSV file from the GitHub repo
    url = 'https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv'
    response = requests.get(url)
    data = StringIO(response.text)
    print("Data fetched.")

    # Read the CSV data into a DataFrame
    print("Reading data into DataFrame...")
    df = pd.read_csv(data)
    print("Data read into DataFrame.")

    # Select key columns
    print("Selecting key columns...")
    key_columns = ['time', 'BlkCnt', 'FeeTotNtv', 'FeeTotUSD', 'HashRate', 'PriceUSD']
    df = df[key_columns]
    print("Key columns selected.")

    # Convert the 'time' column to datetime and set it as the index
    print("Converting 'time' column to datetime and setting as index...")
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    print("Index set.")

    # Create a 'blockHeight' column as the cumulative sum of 'BlkCnt'
    print("Creating 'blockHeight' column...")
    df['blockHeight'] = df['BlkCnt'].cumsum()
    print("'blockHeight' column created.")

    # Function to calculate block reward based on the block height
    def get_block_reward(height, halving_interval=210000, initial_reward=50):
        halvings = height // halving_interval
        if halvings >= 64:
            return 0  # Subsidy drops to 0 after 64 halvings
        return initial_reward / (2 ** halvings)

    # Apply the get_block_reward function to create the 'BlockReward' column
    print("Calculating 'BlockReward' column...")
    df['BlockReward'] = df['blockHeight'].apply(get_block_reward)
    print("'BlockReward' column calculated.")

    # Calculate the BlockReward in USD by multiplying with 'PriceUSD'
    df['BlockRewardUSD'] = df['BlockReward'] * df['PriceUSD']

    # Calculate 'MinerRevenue' in BTC
    df['MinerRevenue'] = (df['BlkCnt'] * df['BlockReward']) + df['FeeTotNtv']

    # Calculate 'MinerRevenueUSD' in USD
    df['MinerRevenueUSD'] = df['MinerRevenue'] * df['PriceUSD']

    # Calculate 'HashPrice' in BTC
    df['HashPrice'] = df['MinerRevenue'] / df['HashRate']

    # Calculate 'HashPriceUSD' in USD
    df['HashPriceUSD'] = df['MinerRevenueUSD'] / df['HashRate']

    # Only when we have fee data onwards/ Last date is nan
    print("Filtering data from 2011-01-01 onwards...")
    df = df.loc['2011-01-01':]

    # Remove rows with NaN values in critical columns
    print("Dropping rows with NaN values in critical columns...")
    df = df.dropna(subset=['BlkCnt', 'FeeTotNtv', 'PriceUSD'])
    print("Data cleaning complete.")

    return df

print("Starting script...")

df = get_data()
print("Data retrieved and processed.")

# Call the function with desired parameters
print("Calling plot_daily_rewards_forecast function...")
plot_daily_rewards_forecast(df, window=365, forecast_years=25, seasonal_periods=365)
print("Script completed.")