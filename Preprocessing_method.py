import pandas as pd #https://pandas.pydata.org/
import numpy as np #https://numpy.org/doc/2.1/index.html
import rampy as rp #https://charlesll.github.io/rampy/html/firststeps.html
import os #https://docs.python.org/3/library/os.html
import matplotlib.pyplot as plt
import pybaselines 
from pybaselines import Baseline
from scipy.signal import savgol_filter  
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm


def shift_correction(reference_file, input_file, output_file):
    
    reference_df = pd.read_csv(reference_file)
    reference_df = reference_df.sort_values(reference_df.columns[0], ascending=True).reset_index(drop=True)

    input_df = pd.read_csv(input_file)
    input_df = input_df.sort_values(input_df.columns[0], ascending=True).reset_index(drop=True)

    reference_df.columns = reference_df.columns.str.strip()
    input_df.columns = input_df.columns.str.strip()

    reference_shift = reference_df['Raman_Shift']
    reference_intensity = reference_df['Intensity']
    input_shift = input_df['Raman_Shift']
    input_intensity = input_df['Intensity']
    corrected_shifts = []

    min_range = 200
    max_range = 2500

    corrected_x = np.empty(shape=0, dtype=float)  
    corrected_y = np.empty(shape=0, dtype=float)  


    for i in range(len(reference_shift)):
        if min_range <= reference_shift[i] <= max_range:
            func = None
            for j in range(len(input_shift)-1):
                if  input_shift[j] <= reference_shift[i] <= input_shift[j+1]:
                    func = interp1d(input_shift[j:j+2], input_intensity[j:j+2], kind='linear', fill_value='extrapolate')
            if func != None:
                corrected_y = np.append(corrected_y, func(reference_shift[i]))
                corrected_x = np.append(corrected_x, reference_shift[i])

  
    corrected_df = pd.DataFrame({
        'Raman_Shift': corrected_x,  
        'Intensity': corrected_y
    })

   
    shift_corrected_file = output_file.replace(".csv", "_corrected.csv")
    corrected_df.to_csv(shift_corrected_file, index=False)
    return corrected_df 
   

def baseline_correction(intensity_data):
   
    smoothed_intensity = savgol_filter(intensity_data, window_length=50, polyorder=10)

    baseline = Baseline () 
    baseline_only, _ = baseline.penalized_poly(
        smoothed_intensity, poly_order=50, tol=0.001, max_iter=250, weights=None, 
        cost_function='asymmetric_truncated_quadratic', threshold=None, 
        alpha_factor=0.99, return_coef=False
        )
    corrected_signal = abs(smoothed_intensity - baseline_only) 
    return corrected_signal
   


def Full_Correction(input_file, reference_file, output_file):
 
    input_data = pd.read_csv(input_file) 

    input_data.columns = input_data.columns.str.strip() 
    corrected_intensity = baseline_correction(input_data['Intensity'].values)
    input_data['Intensity'] = corrected_intensity

    normalized_y = rp.normalise(y=input_data['Intensity'].values, x=input_data['Raman_Shift'].values, method='intensity')
    input_data['Intensity'] = normalized_y

    
    baseline_corrected_file = output_file.replace(".csv", "_baseline.csv")
    input_data.to_csv(baseline_corrected_file, index=False)
       
    shift_correction(reference_file, baseline_corrected_file, output_file)
   
    

def batch_process(input_folder, reference_file, output_folder): 
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_folder, filename)
             
            output_file = os.path.join(output_folder, f"{filename}")
            
            Full_Correction(input_file, reference_file, output_file)
             

input_folder =  # input folder 
reference_file =  # reference file 
output_folder = # outprint folder 

batch_process(input_folder, reference_file, output_folder)
