"""
created 9/20/17

author: adam

file to manage output of network
need to create pyplot as well as excel output
"""
import matplotlib.pyplot as plt

# TODO write function to create pyplot
def create_pyplot_line(output, x_label, y_label):
    plt.plot(output)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# TODO write function to create other plot
# TODO write function to create excel file
def create_excel_file(output):
    wb = Workbook()
    for output_item in output:
