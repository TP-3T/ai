"""
Climate Data Normalization Script
Uses Min-Max scaling to normalize climate prediction datasets to [0, 1] range.
Implements a Pipeline approach for reusability.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
import os


class ClimateDataNormalizer:
    """
    A reusable normalizer for climate prediction datasets using Min-Max scaling.
    Implements a Pipeline approach for consistency across training and testing data.
    """
    
    def __init__(self):
        """Initialize the normalizer with a MinMaxScaler pipeline."""
        # Create a pipeline with MinMaxScaler
        self.pipeline = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
        self.feature_columns = None
        self.excluded_columns = ['ROW_ID']  # Columns to exclude from normalization
        
    def fit_transform(self, df, exclude_cols=None):
        """
        Fit the scaler on the data and transform it.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to normalize
        exclude_cols : list, optional
            Additional columns to exclude from normalization
            
        Returns:
        --------
        pandas.DataFrame
            Normalized dataframe
        """
        if exclude_cols:
            self.excluded_columns.extend(exclude_cols)
        
        # Identify columns to normalize (all except excluded ones)
        self.feature_columns = [col for col in df.columns 
                               if col not in self.excluded_columns]
        
        # Create a copy to avoid modifying original
        df_normalized = df.copy()
        
        # Fit and transform the feature columns
        df_normalized[self.feature_columns] = self.pipeline.fit_transform(
            df[self.feature_columns]
        )
        
        return df_normalized
    
    def transform(self, df):
        """
        Transform data using the already fitted scaler.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to normalize
            
        Returns:
        --------
        pandas.DataFrame
            Normalized dataframe
        """
        if self.feature_columns is None:
            raise ValueError("Pipeline must be fitted before transform. Use fit_transform first.")
        
        # Create a copy to avoid modifying original
        df_normalized = df.copy()
        
        # Transform using the fitted pipeline
        df_normalized[self.feature_columns] = self.pipeline.transform(
            df[self.feature_columns]
        )
        
        return df_normalized
    
    def inverse_transform(self, df):
        """
        Convert normalized data back to original scale.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The normalized dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Denormalized dataframe
        """
        if self.feature_columns is None:
            raise ValueError("Pipeline must be fitted before inverse_transform.")
        
        # Create a copy
        df_denormalized = df.copy()
        
        # Inverse transform
        df_denormalized[self.feature_columns] = self.pipeline.inverse_transform(
            df[self.feature_columns]
        )
        
        return df_denormalized
    
    def save_pipeline(self, filepath):
        """Save the fitted pipeline for future use."""
        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline saved to: {filepath}")
    
    @staticmethod
    def load_pipeline(filepath):
        """Load a previously saved pipeline."""
        pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from: {filepath}")
        return pipeline


def main():
    """Main function to normalize climate datasets."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths relative to script location
    training_file = os.path.join(script_dir, 'climate_prediction_training.csv')
    testing_file = os.path.join(script_dir, 'climate_prediction_testing.csv')
    
    output_dir = os.path.join(script_dir, 'normalized_data')
    os.makedirs(output_dir, exist_ok=True)
    
    normalized_training_file = os.path.join(output_dir, 'climate_prediction_training_normalized.csv')
    normalized_testing_file = os.path.join(output_dir, 'climate_prediction_testing_normalized.csv')
    pipeline_file = os.path.join(output_dir, 'climate_normalization_pipeline.pkl')
    
    print("=" * 70)
    print("Climate Data Normalization using Min-Max Scaling")
    print("=" * 70)
    
    # Load datasets
    print("\n1. Loading datasets...")
    df_train = pd.read_csv(training_file, encoding='utf-8-sig')
    df_test = pd.read_csv(testing_file, encoding='utf-8-sig')
    
    print(f"   - Training data shape: {df_train.shape}")
    print(f"   - Testing data shape: {df_test.shape}")
    
    # Display original data statistics
    print("\n2. Original data statistics (first 5 features):")
    print("-" * 70)
    print(df_train.iloc[:, 1:6].describe())
    
    # Initialize normalizer
    print("\n3. Initializing normalizer pipeline...")
    normalizer = ClimateDataNormalizer()
    
    # Fit on training data and transform
    print("\n4. Fitting scaler on training data and transforming...")
    df_train_normalized = normalizer.fit_transform(df_train)
    
    # Transform testing data using the same scaler
    print("\n5. Transforming testing data using fitted scaler...")
    df_test_normalized = normalizer.transform(df_test)
    
    # Display normalized data statistics
    print("\n6. Normalized data statistics (first 5 features):")
    print("-" * 70)
    print(df_train_normalized.iloc[:, 1:6].describe())
    
    # Verify normalization range
    print("\n7. Verifying normalization range [0, 1]:")
    print("-" * 70)
    feature_cols = [col for col in df_train_normalized.columns if col != 'ROW_ID']
    min_val = df_train_normalized[feature_cols].min().min()
    max_val = df_train_normalized[feature_cols].max().max()
    print(f"   - Minimum value in normalized data: {min_val:.6f}")
    print(f"   - Maximum value in normalized data: {max_val:.6f}")
    
    # Save normalized datasets
    print("\n8. Saving normalized datasets...")
    df_train_normalized.to_csv(normalized_training_file, index=False)
    df_test_normalized.to_csv(normalized_testing_file, index=False)
    print(f"   ✓ Training data saved to: {normalized_training_file}")
    print(f"   ✓ Testing data saved to: {normalized_testing_file}")
    
    # Save the pipeline for future use
    print("\n9. Saving normalization pipeline...")
    normalizer.save_pipeline(pipeline_file)
    
    # Demonstrate inverse transform
    print("\n10. Testing inverse transform (first 3 rows):")
    print("-" * 70)
    df_test_denormalized = normalizer.inverse_transform(df_test_normalized.head(3))
    print("Original values (CO2, TEMP, GMSL):")
    print(df_test.head(3)[['CO2 (ppm)', 'TEMP (deg C)', 'Absolute GMSL (mm) relative to Jan 1950']])
    print("\nDenormalized values (CO2, TEMP, GMSL):")
    print(df_test_denormalized[['CO2 (ppm)', 'TEMP (deg C)', 'Absolute GMSL (mm) relative to Jan 1950']])
    
    print("\n" + "=" * 70)
    print("Normalization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()