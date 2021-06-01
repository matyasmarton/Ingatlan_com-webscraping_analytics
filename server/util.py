# -*- coding: utf-8 -*-

import json
import pickle
import numpy as np


__locations = None
__data_columns = None
__model = None



def get_estimated_price(location,sqm,rooms):
    try:
        loc_index = __data_columns.index(location)
    except:
        loc_index = -1
    
    
    x = np.zeros(len(__data_columns))
    x[0] = sqm
    x[1] = rooms
    # locate the specific location index
    if loc_index >= 0:
        # make that element 1, remaining elements 0
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations
    
    with open('./artifacts/columns test2.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[2:-1]
        
# first 3 columns are sqft, bath, bhk  
        
        
    global __model  
    if __model is None:
        with open('./artifacts/budapest_real_estate_price_model.pickle','rb') as f:
            __model = pickle.load(f)
    print('Loading saved artifacts...done')
    
def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    print(len(__data_columns))
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price(14, 55, 2.5))
    
