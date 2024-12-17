#import module from global env
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import yaml
import time
from datetime import datetime

#import module from this projects
from src.utils import get_current_datetime
from models.discharge.model_ml1 import inference_ml1
from models.inundation.model_ml2 import inference_ml2
from src.data_ingesting import get_input_ml1, get_input_ml1_hujan_max
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable

# Load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Define the input model
class PredictionInput(BaseModel):
    t0: str

app = FastAPI()
def do_prediction(t0=None):
    """
    Run the prediction pipeline for both ML1 (discharge) and ML2 (inundation) models.

    This function performs the following steps:
    
    1. **Data Ingestion**: 
       - Ingest precipitation data for the last 72 hours from a specified path and processes the data into a format suitable for model ML1.
       - The function also accepts a time parameter `t0`, which is used to specify the starting time for the data processing.
    
    2. **ML1 Prediction**:
       - Predicts discharge (debit) using the input data for ML1 (discharge model).
       - The last 24 hours of predicted debit are selected for input into the next model (ML2).
    
    3. **ML2 Prediction**:
       - Predicts inundation levels using the output from ML1 (discharge).
       - The input to ML2 is reshaped into the required format before prediction.

    4. **Post-Processing**:
       - The predictions from both models are bundled into dictionaries, which include associated date information, precipitation data, and predictions from both models.
       - The output is then converted into a JSON-compatible format to ensure all elements are serializable.
    
    5. **Return**:
       - The function returns a dictionary containing:
         - Start and finish times of the prediction.
         - Precipitation information and sources.
         - Predictions for both discharge (ML1) and inundation (ML2).
    
    Args:
        t0 (str): A time parameter used to specify the starting time for data ingestion. If None, then using current datetime
    
    Returns:
        dict: A dictionary containing the prediction results and metadata, including:
              - Prediction Time Start and Finished.
              - Precipitation information and sources.
              - ML1 (discharge) and ML2 (inundation) prediction outputs.
    """
    # Get configuration to run the system
    start_run_time = get_current_datetime()
    if t0:
        t0 = datetime.strptime(t0, '%Y-%m-%d %H:%M:%S')

    config_path = 'config.yaml'
    config = load_config(config_path)
    input_size_ml2 = config['model']['input_size_ml2']

    #1. Ingest data hujan
    t_start_ingest = time.time()
    path_hujan_hist_72jam = config['data_processing']['path_hujan_hist_72jam']
    input_ml1, ch_wilayah, date_list, data_information, data_name_list = get_input_ml1(t0=t0,config=config)
    input_ml1, ch_wilayah = get_input_ml1_hujan_max() #hanya untuk menampilkan possible banjur terbesar
    t_end_ingest = time.time()

    print(f"Succesfully ingesting the data: {t_end_ingest-t_start_ingest}s")

    #2. Predict debit using ML1
    t_start_ml1 = time.time()
    debit = inference_ml1(input_ml1,config)
    #debit = debit + 500 #Ini cuma percobaan kalau debitnya dibesarkan, banjirnya gimana
    input_ml2 = debit[-input_size_ml2:].view(1,input_size_ml2,1)
    t_end_ml1 = time.time()

    print(f"Succesfully inference ml1: {t_end_ml1-t_start_ml1}s")

    #3. Predict inundation using ML2
    t_start_ml2 = time.time()
    genangan = inference_ml2(input_ml2)
    t_end_ml2 = time.time()
    print(f"Succesfully inference ml2: {t_end_ml2-t_start_ml2}s")
    end_run_time = get_current_datetime()

    #4. Bundle output
    dates, dict_output_ml1 = output_ml1_to_dict(dates=date_list, output_ml1=debit.tolist(), precipitation=ch_wilayah)
    dict_output_ml2 = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=genangan)

    output = {"Prediction Time Start": str(start_run_time), 
            "Prediction Time Finished": str(end_run_time),
            "precip information": data_information,
            "precip source":data_name_list,
            "precip date": date_list,
            "Prediction Output ML1": dict_output_ml1,
            "Prediction Output ML2": dict_output_ml2}
    
    output = ensure_jsonable(output)
    return output

# @app.post("/predict")
# async def predict(input_data: PredictionInput):
#     output = do_prediction(t0=input_data.t0)
#     return output

@app.post("/predict")
async def predict():
    output = do_prediction(t0=None)
    return output

# Run the application using the following command:
# uvicorn app2:app --reload
