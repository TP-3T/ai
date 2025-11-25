"""
Use trained GAM models to predict future temperature and sea level.
Enhanced with cProfile and psutil for performance monitoring.
"""

import pickle
import pandas as pd
import numpy as np
import cProfile
import pstats
import io
from memory_profiler import profile
import psutil
import time
import os

from constants.data_file_paths import GAM_TEMP_MODEL_FILENAME, GAM_SEA_LEVEL_MODEL_FILENAME
from constants.climate_prediction_dataset_cols import CO2, TEMP, SEA_LVL


class PerformanceMonitor:
    """Monitor CPU and memory usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_cpu_percent = None
        self.start_memory = None
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_cpu_percent = self.process.cpu_percent(interval=0.1)
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop_and_report(self, label="Operation"):
        """Stop monitoring and print report."""
        end_time = time.time()
        end_cpu_percent = self.process.cpu_percent(interval=0.1)
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        elapsed_time = end_time - self.start_time
        avg_cpu = (self.start_cpu_percent + end_cpu_percent) / 2
        memory_diff = end_memory - self.start_memory
        
        print(f"\n{'='*70}")
        print(f"⚡ PERFORMANCE METRICS - {label}")
        print(f"{'='*70}")
        print(f"  ⏱️  Execution Time: {elapsed_time:.4f} seconds")
        print(f"  💻 CPU Usage: {avg_cpu:.2f}%")
        print(f"  🧠 Memory Used: {end_memory:.2f} MB")
        print(f"  📊 Memory Change: {memory_diff:+.2f} MB")
        print(f"{'='*70}\n")
        
        return {
            'time': elapsed_time,
            'cpu': avg_cpu,
            'memory': end_memory,
            'memory_change': memory_diff
        }


@profile
def load_models():
    """Load the trained GAM models from pickle files."""
    print("Loading trained models...")
    
    with open(GAM_TEMP_MODEL_FILENAME, 'rb') as f:
        temp_model = pickle.load(f)
    print(f"✅ Temperature model loaded from {GAM_TEMP_MODEL_FILENAME}")
    
    with open(GAM_SEA_LEVEL_MODEL_FILENAME, 'rb') as f:
        sea_level_model = pickle.load(f)
    print(f"✅ Sea level model loaded from {GAM_SEA_LEVEL_MODEL_FILENAME}")
    
    return temp_model, sea_level_model


@profile
def predict_climate(temp_model, sea_level_model, co2_ppm, current_temp, current_sea_level):
    """
    Predict future climate conditions based on current state.
    
    Args:
        temp_model: Trained temperature GAM model
        sea_level_model: Trained sea level GAM model
        co2_ppm: Current CO2 concentration in parts per million (ppm)
        current_temp: Current global average temperature in °C
        current_sea_level: Current global average sea level in mm
        
    Returns:
        Tuple of (predicted_temperature, predicted_sea_level)
    """
    
    # Create input features as a 2D array (models expect this format)
    # Order: [CO2, Temperature, Sea Level]
    features = np.array([[co2_ppm, current_temp, current_sea_level]])
    
    # Make predictions
    future_temp = temp_model.predict(features)[0]
    future_sea_level = sea_level_model.predict(features)[0]
    
    return future_temp, future_sea_level


@profile
def predict_multiple_years(temp_model, sea_level_model, 
                          initial_co2, initial_temp, initial_sea_level,
                          co2_increase_per_year, num_years):
    """
    Predict climate conditions for multiple years into the future.
    
    Args:
        temp_model: Trained temperature GAM model
        sea_level_model: Trained sea level GAM model
        initial_co2: Starting CO2 concentration (ppm)
        initial_temp: Starting temperature (°C)
        initial_sea_level: Starting sea level (mm)
        co2_increase_per_year: Expected CO2 increase per year (ppm/year)
        num_years: Number of years to predict
        
    Returns:
        DataFrame with predictions for each year
    """
    
    predictions = []
    
    # Current state
    current_co2 = initial_co2
    current_temp = initial_temp
    current_sea_level = initial_sea_level
    
    print(f"\nPredicting {num_years} years into the future...")
    print(f"Starting conditions:")
    print(f"  CO2: {current_co2:.2f} ppm")
    print(f"  Temperature: {current_temp:.2f} °C")
    print(f"  Sea Level: {current_sea_level:.2f} mm")
    print(f"  CO2 increase per year: {co2_increase_per_year:.2f} ppm/year\n")
    
    for year in range(1, num_years + 1):
        # Predict next year's conditions based on current state
        future_temp, future_sea_level = predict_climate(
            temp_model, sea_level_model,
            current_co2, current_temp, current_sea_level
        )
        
        # Increase CO2 for next year
        current_co2 += co2_increase_per_year
        
        # Update current conditions with predictions (for next iteration)
        current_temp = future_temp
        current_sea_level = future_sea_level
        
        # Store prediction
        predictions.append({
            'Year': year,
            'CO2 (ppm)': current_co2,
            'Temperature (°C)': current_temp,
            'Sea Level (mm)': current_sea_level,
            'Temp Change (°C)': current_temp - initial_temp,
            'Sea Level Rise (mm)': current_sea_level - initial_sea_level
        })
    
    return pd.DataFrame(predictions)


def run_profiled_predictions():
    """Main function with profiling enabled."""
    
    # Overall performance monitor
    overall_monitor = PerformanceMonitor()
    overall_monitor.start()
    
    # Track metrics for each section
    metrics = {}
    
    # ========== MODEL LOADING ==========
    print("\n" + "=" * 70)
    print("📦 LOADING MODELS")
    print("=" * 70)
    
    load_monitor = PerformanceMonitor()
    load_monitor.start()
    temp_model, sea_level_model = load_models()
    metrics['model_loading'] = load_monitor.stop_and_report("Model Loading")
    
    # ========== EXAMPLE 1: Single Year Prediction ==========
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Year Prediction")
    print("=" * 70)
    
    single_monitor = PerformanceMonitor()
    single_monitor.start()
    
    # Example: Predict one year ahead
    co2 = 500  # Current CO2 level (ppm)
    temp = 14.1  # Current global temperature (°C)
    sea_level = 0.0  # Current sea level (mm above baseline)
    
    future_temp, future_sea_level = predict_climate(
        temp_model, sea_level_model, co2, temp, sea_level
    )
    
    print(f"\nCurrent conditions:")
    print(f"  CO2: {co2} ppm")
    print(f"  Temperature: {temp} °C")
    print(f"  Sea Level: {sea_level} mm")
    
    print(f"\nPredicted next year:")
    print(f"  Temperature: {future_temp:.2f} °C (change: {future_temp - temp:+.2f} °C)")
    print(f"  Sea Level: {future_sea_level:.2f} mm (rise: {future_sea_level - sea_level:+.2f} mm)")
    
    metrics['single_prediction'] = single_monitor.stop_and_report("Single Year Prediction")
    
    # ========== EXAMPLE 2: 10-Year Prediction ==========
    print("\n" + "=" * 70)
    print("EXAMPLE 2: 10-Year Prediction (ALL YEARS SHOWN)")
    print("=" * 70)
    
    multi_monitor = PerformanceMonitor()
    multi_monitor.start()
    
    # Predict 10 years into the future
    predictions_df = predict_multiple_years(
        temp_model, sea_level_model,
        initial_co2=420.0,
        initial_temp=14.8,
        initial_sea_level=100.0,
        co2_increase_per_year=2.5,  # Assume 2.5 ppm increase per year
        num_years=10
    )
    
    metrics['multi_year_prediction'] = multi_monitor.stop_and_report("10-Year Prediction")
    
    print("\n📊 Year-by-Year Predictions:")
    print("=" * 70)
    # Display all rows with better formatting
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)      # Don't wrap columns
    pd.set_option('display.float_format', '{:.2f}'.format)  # 2 decimal places
    print(predictions_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("📈 SUMMARY - FINAL CONDITIONS (Year 10):")
    print("=" * 70)
    final_row = predictions_df.iloc[-1]
    print(f"  CO2: {final_row['CO2 (ppm)']:.2f} ppm")
    print(f"  Temperature: {final_row['Temperature (°C)']:.2f} °C")
    print(f"  Sea Level: {final_row['Sea Level (mm)']:.2f} mm")
    print(f"  Total temperature increase: {final_row['Temp Change (°C)']:.2f} °C")
    print(f"  Total sea level rise: {final_row['Sea Level Rise (mm)']:.2f} mm")
    
    # Optional: Save predictions to CSV
    output_file = "climate_predictions_10_years.csv"
    predictions_df.to_csv(output_file, index=False)
    print(f"\n💾 Predictions saved to {output_file}")
    
    # ========== OVERALL PERFORMANCE SUMMARY ==========
    metrics['total'] = overall_monitor.stop_and_report("TOTAL EXECUTION")
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Operation':<25} {'Time (s)':<12} {'CPU %':<10} {'Memory (MB)':<12}")
    print("-" * 70)
    for operation, data in metrics.items():
        print(f"{operation.replace('_', ' ').title():<25} {data['time']:<12.4f} {data['cpu']:<10.2f} {data['memory']:<12.2f}")
    print("=" * 70)
    
    print("\n✅ PREDICTION COMPLETE WITH PROFILING!")
    print("=" * 70)


if __name__ == "__main__":
    # Create profiler
    profiler = cProfile.Profile()
    
    print("\n" + "=" * 70)
    print("🚀 STARTING PROFILED EXECUTION")
    print("=" * 70)
    
    # Run with profiling
    profiler.enable()
    run_profiled_predictions()
    profiler.disable()
    
    # Print cProfile statistics
    print("\n" + "=" * 70)
    print("📈 cProfile DETAILED STATISTICS")
    print("=" * 70)
    
    # Create string buffer for stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time and print top 20 functions
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    print(s.getvalue())
    
    # Also sort by total time
    print("\n" + "=" * 70)
    print("📊 TOP 20 FUNCTIONS BY TOTAL TIME")
    print("=" * 70)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats(20)
    print(s.getvalue())
    
    # Save detailed profile to file
    profile_output = "gam_predict_profile.stats"
    profiler.dump_stats(profile_output)
    print(f"\n💾 Detailed profile saved to {profile_output}")
    print("   (View with: python -m pstats {profile_output})")