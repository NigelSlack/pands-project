#---------------------------------------------------------------------------------------------------#
# Program      : iris_dataset.py
# Author       : Nigel Slack
# Language     : python
#
# Function     : Use a GUI to display the data and simple numerical and graphical statistical 
#                analyses of the 'well known' Iris dataset (published by statistician Ronald Fisher
#                in his 1936 paper 'The use of multiple measurements in taxonomic problems)'.
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
# 17/03/2019 NS Use indices for showing data, stats and correlations, and include separate 
#               correlations for each species. Append median and variance to 'describe' output
#               and use this for the stats. Re-arrange buttons and use destroy when quitting so
#               that Quit only needs to be pressed once. Help text expanded a little. Allow for
#               'Help' run time argument.
#
#----------------------------------------------------------------------------------------------------#

#ref https://www.kaggle.com/gopaltirupur/iris-data-analysis-and-machine-learning-python
import sys
import pandas as pd
import numpy as np
import matplotlib
#ref https://stackoverflow.com/questions/47553142/import-issue-with-matplotlib-and-pyplot-python-tkinter
matplotlib.use("TkAgg")
from functools import partial
from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *

hlptext = """\n Display data and statistical analyses of the well known 'Iris data set'.
 The data provided by a collection of fifty samples each of three species of
 Iris collected by biologist Edgar Anderson were analysed in a paper in 1936
 by statistician Ronald Fisher. Since then this data set has been widely used for 
 testing statistical techniques.
 This program provides an easy to use GUI to examine the data itself, and simple statistical
 features of the data.
 Syntax : python iris_dataset.py [help]"""

def helptext():  
  print(hlptext)  
      
# Use the python 'sys' module to check for run time arguments. 
# If the user input 'help' output help text, otherwise tell them no arguments are required.
# ref https://stackabuse.com/command-line-arguments-in-python/
if len(sys.argv)-1:
  if sys.argv[1].upper() == "HELP":   
    helptext()
  else:
    print(" \nNo run time arguments required")
  # end-if  
# end-if 

df = pd.read_csv('irisdata.csv')
#ref https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
# number of rows given by len(df.index)
# ref https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe

# Add median and variance to the stats provided by describe
#ref https://stackoverflow.com/questions/38545828/pandas-describe-by-additional-parameters
def describe(df):
    return pd.concat([df.describe().T,
                      df.median().rename('median'),
                      df.var().rename('variance'),
                     ], axis=1).T
                     
def describex(df,ix):
    return pd.concat([df[df[columnNames[4]]==species[ix]].describe().T,
                      df[df[columnNames[4]]==species[ix]].median().rename('median'),
                      df[df[columnNames[4]]==species[ix]].var().rename('variance'),                      
                     ], axis=1).T                     
                     
pd.set_option('display.max_rows',len(df.index))
statall=describe(df)
sl = str(df['species'].unique()).strip('[').strip(']')

columnNames = list(df.head(0)) 
species = (df['species'].unique())
co = ['r','b','g']
dx,stat =([],[])
dx.append(df)

#ref https://stackoverflow.com/questions/52350313/python-for-loop-create-a-list
for i in range (3):
  dx.append(df[df[columnNames[4]]==species[i]])
  stat.append(describex(df[df[columnNames[4]]==species[i]],i))

# Display correlation between columns
# ref https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3
def plot_correlation(ix):
# ref https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe 
  fig = plt.figure()
  corr = dx[ix].iloc[:,0:4].corr()
  ax = fig.add_subplot(111)
  cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
  fig.colorbar(cax)
  ticks = np.arange(0,len(dx[ix].iloc[:,0:4].columns),1)
  ax.set_xticks(ticks)
  plt.xticks(rotation=0)
  ax.set_yticks(ticks)
  ax.set_xticklabels(dx[ix].iloc[:,0:4].columns)
  ax.set_yticklabels(dx[ix].iloc[:,0:4].columns)
  if ix:
    atitle = 'Pearson correlation for Sepal/Petal length for ' + species[(ix-1)]
  else:
    atitle = 'Pearson correlation for Sepal/Petal length for ' + sl
  ax.set(title=atitle)
  plt.show()
  
def print_data():
  print("\n",df,"\n")
  
# ref https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas/17071908
# print(df.loc[df['species'] == species[0]])

# ref https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas/17071908
# print(df.loc[(df[columnNames[0]] >= 4.0) & (df[columnNames[0]] <= 5.0)])

# ref https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# print(df.groupby(columnNames[4]).size())

def show_species(ix):
  print("\n",dx[(ix+1)],"\n")
  
# ref https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
def plot_scatter():
  # plt.figure()
  fig,ax=plt.subplots(1,2,figsize=(17, 9))
  for i in range (3):
    dx[i].plot(x="sepal_length", y="sepal_width", kind="scatter",ax=ax[0],label=species[i],color=co[i])
    dx[i].plot(x="petal_length", y="petal_width", kind="scatter",ax=ax[1],label=species[i],color=co[i])     
  ax[0].set(title='Sepal comparison ', ylabel='sepal-width')
  ax[1].set(title='Petal Comparison',  ylabel='petal-width')
  ax[0].legend()
  ax[1].legend()
  plt.show()
  plt.close()
  
# ref https://stackoverflow.com/questions/49970309/how-do-i-calculate-the-mean-of-each-species-of-the-iris-data-set-in-python
# ref https://www.tutorialspoint.com/python_pandas/python_pandas_descriptive_statistics.htm
def print_stats():
  print("\nStats for ",sl)
  print(statall)
  print(" ")
  
def print_statsx(ix):
  print("\nStats for ",species[ix])
  print(stat[ix])
  print(" ")
  
def close_all():
  master.quit()
  master.destroy()
        
master = Tk()

Button(master, text='Show all data', command=print_data).grid(row=0, column=0, sticky=W, pady=4)
for i in range(3):
  Button(master, text='Data for ' + species[i], command=partial(show_species,i)).grid(row=0, column=(i+1), sticky=W, pady=4)

Button(master, text='Show overall stats', command=print_stats).grid(row=1, column=0, sticky=W, pady=4)
for i in range(3):
  Button(master, text='Stats for ' + species[i], command= partial(print_statsx,i)).grid(row=1,column=(i+1),sticky=W,pady=4)
  
Button(master, text='Overall correlations', command= partial(plot_correlation,0)).grid(row=2, column=0, sticky=W, pady=4)
for i in range(1,4):
  Button(master, text='Correlations for ' + species[(i-1)], command= partial(plot_correlation,i)).grid(row=2, column=(i), sticky=W, pady=4)
  
Button(master, text='Scatter graphs', command=plot_scatter).grid(row=3, column=0, sticky=W, pady=4)
# ref https://stackoverflow.com/questions/6920302/how-to-pass-arguments-to-a-button-command-in-tkinter/22290388

Button(master, text='Help', command=helptext).grid(row=4, column=0, sticky=W, pady=4)
Button(master, text='Quit', command=close_all).grid(row=5, column=0, sticky=W, pady=4)

mainloop()