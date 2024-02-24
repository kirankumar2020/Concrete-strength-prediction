import pickle
import json
import numpy as np
import pandas as pd


def get_predicted_strength(Cement, Blast_Furnace_Slag, Fly_Ash, Water, Superplasticizer, Coarse_Aggregate, Fine_Aggregate, Age):
    mobile_path =r"D:\PDS\Flask\concrete_data-2\Concrete....RandomForest_reg_model.pkl"

    with open("Concrete....RandomForest_reg_model.pkl", 'rb') as f:
        model = pickle.load(f)



        test_array = np.array([Cement, Blast_Furnace_Slag, Fly_Ash, Water, Superplasticizer, Coarse_Aggregate, Fine_Aggregate, Age], ndmin = 2)
        # print(test_array)

        predicted_strength = model.predict(test_array)[0]
        predicted_strength = round(predicted_strength, 2)
        # print("predicted_strength :",predicted_strength)

        return predicted_strength