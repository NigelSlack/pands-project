#---------------------------------------------------------------------------------------------#
# Program      : iris_dataset.py
# Author       : Nigel Slack
# Language     : python
#
# Function     : Perform a statistical and graphical analysis of the Iris dataset pulished by
#                Ronald Fisher in his 1936 paper 'The use of multiple measurements in taxonomic 
#                problems'
#
# Syntax       : python iris_dataset.py [help]
#
# Dependencies : System 'sys' module  - proc runtime args
#                System 'math' module  - to get square root
# Arguments    : 'Help' can be accepted as a run time argument
#                None
#
# Versions     :
# 14/03/2019 NS Initial
#
#-----------------------------------------------------------------------------------------------#

#ref https://www.kaggle.com/gopaltirupur/iris-data-analysis-and-machine-learning-python
import pandas as pd
import sys
import numpy as np
import matplotlib

#ref https://stackoverflow.com/questions/47553142/import-issue-with-matplotlib-and-pyplot-python-tkinter
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# import seaborn as sns
from tkinter import *
# sns.set(color_codes=True)

df = pd.read_csv('irisdata.csv')


#ref https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
# number of rows given by len(df.index)
# ref https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
pd.set_option('display.max_rows',len(df.index))

def print_data():
  print(df)

# df.head()
# df.info()
# df['species'].unique()
# print(df.describe())
# input("Press Enter to continue...")

species = (df['species'].unique())
# print(df)
# print(species)

# ref https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas/17071908
# print(df.loc[df['species'] == species[0]])

columnNames = list(df.head(0)) 
# print(columnNames)

# ref https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas/17071908
# print(df.loc[(df[columnNames[0]] >= 4.0) & (df[columnNames[0]] <= 5.0)])

# ref https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# print(df.groupby(columnNames[4]).size())

def show_species_0():
  s1=df[df[columnNames[4]]==species[0]]
  print(s1.describe)

def show_species_1():
  s1=df[df[columnNames[4]]==species[1]]
  print(s1.describe)

def show_species_2():
  s1=df[df[columnNames[4]]==species[2]]
  print(s1.describe)
  
def helptext():  
  print("""This code displays statistical information about the 'Iris data set'.
           In 1936 biologist""")

# ref https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
def plot_scatter():
  plt.figure()
  fig,ax=plt.subplots(1,2,figsize=(17, 9))
  df.plot(x="sepal_length",y="sepal_width",kind="scatter",ax=ax[0],sharex=False,sharey=False,label="sepal",color='r')
  df.plot(x="petal_length",y="petal_width",kind="scatter",ax=ax[1],sharex=False,sharey=False,label="petal",color='b')
  ax[0].set(title='Sepal comparison ', ylabel='sepal-width')
  ax[1].set(title='Petal Comparison',  ylabel='petal-width')
  ax[0].legend()
  ax[1].legend()
  plt.show()
  plt.close()

# ref https://stackoverflow.com/questions/49970309/how-do-i-calculate-the-mean-of-each-species-of-the-iris-data-set-in-python
# ref https://www.tutorialspoint.com/python_pandas/python_pandas_descriptive_statistics.htm
def print_stats():
  print (df.groupby('species').mean())
  print (df.groupby('species').sum())
  print (df.groupby('species').std())
  print (df.groupby('species').median())
  print (df.groupby('species').min())
  print (df.groupby('species').max())
  
master = Tk()

# Need to format the text for the button 'text' content as string format
# https://stackoverflow.com/questions/14510871/curly-braces-showing-in-text-in-python

t0 = ("Show species {}").format(species[0])
t1 = ("Show species {}").format(species[1])
t2 = ("Show species {}").format(species[2])

Button(master, text='Show data set', command=print_data).grid(row=0, column=0, sticky=W, pady=4)
Button(master, text='Show overall stats', command=print_stats).grid(row=1, column=0, sticky=W, pady=4)
Button(master, text='Plot scatter graphs', command=plot_scatter).grid(row=2, column=0, sticky=W, pady=4)
Button(master, text=t0, command=show_species_0).grid(row=3, column=0, sticky=W, pady=4)
Button(master, text=t1, command=show_species_1).grid(row=4, column=0, sticky=W, pady=4)
Button(master, text=t2, command=show_species_2).grid(row=5, column=0, sticky=W, pady=4)
Button(master, text='Help', command=helptext).grid(row=6, column=0, sticky=W, pady=4)
Button(master, text='Quit', command=master.quit).grid(row=7, column=0, sticky=W, pady=4)

mainloop()