# -----------------------------------------------
# data_processing.py
# -----------------------------------------------

# Importing necessary libraries
import pandas as pd              # Pandas: used for loading data and cleaning
import numpy as np               # Numpy: used for numerical operations
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# LabelEncoder: converts categories -> numbers
# StandardScaler: standardizes values (mean=0, std=1)
# MinMaxScaler: scales values to 0–1 range

# ------------------------------------------------
# FUNCTION 1: Load and Clean Data
# ------------------------------------------------
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)  
    # pandas reads the CSV file and converts it into a DataFrame (table format)

    # Removing duplicates
    data = data.drop_duplicates(keep="first")
    # Drop exact duplicate rows – keeps only the first occurrence.
    # This reduces noise and prevents duplicated examples from affecting the model.

    return data  # returns cleaned dataset


# ------------------------------------------------
# FUNCTION 2: Encoding Categorical Variables
# ------------------------------------------------
def encoder(method, dataframe, columns_label, columns_onehot):

    # ---------------------------
    # METHOD 1: LABEL ENCODER
    # ---------------------------
    if method == 'labelEncoder':      
        df_lbl = dataframe.copy()   
        # Make a copy to avoid overwriting original data

        for col in columns_label:
            label = LabelEncoder()  
            # LabelEncoder converts categories → integers
            # Example: ["Male","Female"] → [1,0]

            label.fit(list(dataframe[col].values))
            # Learns unique categories in the column (e.g., "SUV","Sedan","Bus")

            df_lbl[col] = label.transform(df_lbl[col].values)
            # Transforms each category to a number
            # MACHINE LEARNING MODELS need numbers, not text!

        return df_lbl
    
    # ---------------------------
    # METHOD 2: ONE-HOT ENCODER
    # ---------------------------
    elif method == 'oneHotEncoder':
        df_oh = dataframe.copy()

        df_oh = pd.get_dummies(
            data=df_oh,
            prefix='ohe',          # Adds prefix before new columns
            prefix_sep='_',        # Separator in column name 
            columns=columns_onehot,# Columns to encode
            drop_first=True,       # Drops first category to avoid "dummy variable trap"
            dtype='int8'           # Makes new columns memory-efficient
        )
        # One-hot encoding example:
        # Color = ["Red","Blue","Green"]
        # → ohe_Color_Blue (0/1), ohe_Color_Green (0/1)

        return df_oh


# ------------------------------------------------
# FUNCTION 3: SCALING NUMERICAL VARIABLES
# ------------------------------------------------
def scaler(method, data, columns_scaler):

    # ---------------------------
    # METHOD 1: STANDARD SCALER
    # ---------------------------
    if method == 'standardScaler':
        Standard = StandardScaler()
        # StandardScaler converts values so that:
        # mean = 0  
        # standard deviation = 1
        # It is used in models like Linear Regression, SVM, Logistic Regression.

        df_standard = data.copy()
        df_standard[columns_scaler] = Standard.fit_transform(df_standard[columns_scaler])
        # Fit: learn mean & std from data
        # Transform: apply (x - mean)/std
        # Makes all numerical columns comparable in scale

        return df_standard
        
    # ---------------------------
    # METHOD 2: MIN-MAX SCALER
    # ---------------------------
    elif method == 'minMaxScaler':
        MinMax = MinMaxScaler()
        # MinMaxScaler transforms values between 0 and 1
        # Useful for:
        # - Neural Networks
        # - Distance-based models (KNN, KMeans)

        df_minmax = data.copy()
        df_minmax[columns_scaler] = MinMax.fit_transform(df_minmax[columns_scaler])
        # Formula: (x - min) / (max - min)

        return df_minmax
    
    # ---------------------------
    # METHOD 3: LOG TRANSFORMATION
    # ---------------------------
    elif method == 'npLog':
        df_nplog = data.copy()

        # Log transformation example:
        # If column = [1, 10, 100, 1000]
        # log(column) = [0, 2.30, 4.60, 6.90]
        #
        # WHY USE LOG?
        # - Reduces effect of extreme values (outliers)
        # - Helps "skewed" data become more normal
        # - Improves linear model performance

        df_nplog[columns_scaler] = np.log(df_nplog[columns_scaler])
        return df_nplog
    
    return data  # fallback if no method chosen
