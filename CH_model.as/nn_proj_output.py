"""
created 9/20/17

author: adam

file to manage output of network
need to create pyplot as well as excel output
"""
import matplotlib.pyplot as plt
import pandas as pd

# TODO write function to create pyplot
def create_pyplot_line(output, x_label, y_label):
    plt.plot(output)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# TODO write function to create other plot
    
def create_dataframe(input_data):
    df = pd.DataFrame(data=input_data)
    return df

def create_series(input_data, name='attr'):
    srs = pd.Series(data=input_data, name=str(name))
    return srs

# TODO write function to create excel file