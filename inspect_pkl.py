# inspect_pkl.py
import joblib
import sys
import warnings
import xgboost # Need this import for type checking below

# Ignore the specific XGBoost version warning during inspection
warnings.filterwarnings("ignore", message="If you are loading a serialized model")

FILENAME = 'xgb_model.pkl' # Make sure this matches your actual file name

print(f"Attempting to load '{FILENAME}'...")

try:
    # Attempt to load whatever is in the file
    loaded_data = joblib.load(FILENAME)
    print("-" * 30)
    print("File loaded successfully!")
    print(f"Type of loaded data: {type(loaded_data)}") # THIS IS THE KEY OUTPUT
    print("-" * 30)

    if isinstance(loaded_data, dict):
        print("It IS a dictionary. Contents (Keys):")
        print(list(loaded_data.keys()))
        for key, value in loaded_data.items():
            # Print type, and also specifically check if 'pipeline' key holds a Pipeline object
            print(f"  - Key '{key}': Type {type(value)}")
            if key == 'pipeline':
                 # You might need to import Pipeline from sklearn.pipeline for this check
                 from sklearn.pipeline import Pipeline
                 print(f"    Is pipeline key a scikit-learn Pipeline object? {isinstance(value, Pipeline)}")
        print("\nThis file looks CORRECTLY saved!")
    # Explicitly check if it's the XGBRegressor type
    elif isinstance(loaded_data, xgboost.sklearn.XGBRegressor):
        print(">>> It is an XGBRegressor object. <<<")
        print("This confirms the file was NOT saved as the expected dictionary bundle.")
        print("You MUST go back to the training script (train_and_save_model.py) and fix the saving part")
        print("to save the dictionary {'pipeline': ..., 'min_date': ..., 'feature_names_in': ...}")
        print("Make sure you are running the CORRECTED training script and using the NEWLY generated .pkl file.")
    else:
        print(">>> It is NOT a dictionary and NOT a direct XGBRegressor object. <<<")
        print(f"Type found: {type(loaded_data)}. This is unexpected.")
        print("Please double-check the saving process in train_and_save_model.py.")


    print("-" * 30)

except FileNotFoundError:
    print(f"Error: File '{FILENAME}' not found in the current directory.")
    print(f"Make sure '{FILENAME}' is in the same folder as this script.")
except Exception as e:
    print(f"An error occurred while loading or inspecting the file: {e}")
    print("This could be due to severe version incompatibility or file corruption.")

sys.exit(0)