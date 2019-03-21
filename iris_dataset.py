﻿#---------------------------------------------------------------------------------------------------#
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
<<<<<<< HEAD
# 18/03/2019 NS Put scatter plots in separate outputs, with 2 extra ones - petal/sepal len/width 
# 21/03/2019 NS Include photos        
=======
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
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
<<<<<<< HEAD
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy import polyval, stats
from pylab import imread,subplot,imshow,show
=======
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf

hlptext = """\n Display data and statistical analyses of the well known 'Iris data set'.
 The data provided by a collection of fifty samples each of three species of
 Iris collected by biologist Edgar Anderson were analysed in a paper in 1936
 by statistician Ronald Fisher. Since then this data set has been widely used for 
 testing statistical techniques.
<<<<<<< HEAD

=======
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
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
<<<<<<< HEAD

# Turn bold on/off
# ref https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
blx = '\033[0m'
bld = '\033[1m'
=======
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
dx,stat =([],[])
dx.append(df)

#ref https://stackoverflow.com/questions/52350313/python-for-loop-create-a-list
for i in range (3):
  dx.append(df[df[columnNames[4]]==species[i]])
  stat.append(describex(df[df[columnNames[4]]==species[i]],i))
<<<<<<< HEAD
  
# Show photos of each species
def show_photos():
  itext= \
   """   Fig 1 : http://www.cfgphoto.com/photo-55675.htm 
   Fig 2 : https://commons.wikimedia.org/wiki/File:Iris_versicolor_3.jpg 
   Fig 3 : https://commons.wikimedia.org/wiki/File:Iris_virginica_-_NRCS.jpg"""

  # ref https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly  
  fig=plt.figure(figsize=(12,5.5))  
      
  # ref http://www.cfgphoto.com/photo-55675.htm
  # ref https://stackoverflow.com/questions/35286540/display-an-image-with-python
  image = imread('Iris_Setosa.jpg')  # choose image location
  fig.add_subplot(221)
  plt.title('Fig 1 : Iris setosa')
  # ref https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
  plt.axis('off')
  plt.imshow(image)

  # ref https://commons.wikimedia.org/wiki/File:Iris_versicolor_3.jpg
  image = imread('Iris_Versicolor.jpg')  # choose image location
  fig.add_subplot(222)
  plt.title('Fig 2 : Iris versicolor')
  plt.axis('off')
  plt.imshow(image)

  #ref https://commons.wikimedia.org/wiki/File:Iris_virginica_-_NRCS.jpg
  image = imread('Iris_Virginica.jpg')  # choose image location
  ax=fig.add_subplot(223)
  # ref https://matplotlib.org/gallery/text_labels_and_annotations/annotation_demo.html
  ax.annotate('sepal',xy=(0.2, 0.6), xycoords='axes fraction',xytext=(0.2, 0.3), textcoords='axes fraction',
              arrowprops=dict(facecolor='white', shrink=0.05),
              horizontalalignment='right', verticalalignment='top',color='white')
  ax.annotate('petal',xy=(0.5, 0.9), xycoords='axes fraction',xytext=(0.8, 0.9), textcoords='axes fraction',
              arrowprops=dict(facecolor='white', shrink=0.05),
              horizontalalignment='right', verticalalignment='top',color='white')            
  plt.title('Fig 3 : Iris virginica')
  plt.axis('off')
  plt.imshow(image)

  # ref http://www.futurile.net/2016/03/01/text-handling-in-matplotlib/
  plt.figtext(.5, .05, itext, multialignment='left')
  plt.show()
  
#c1=[]  
#c2=[]
#ix=len(dx[1])
#for i in range(ix):
#  c2.append(i)
#  i+=1
#c1 = sorted(list(dx[1]['sepal_length']))  

#plt.plot(c2,c1)
#plt.show()
#plt.close()
#input("stop")
#sys.exit()
          
#iz=3
#ty='petal_length'
# ref https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
#stat, p = shapiro(dx[iz][ty])
#print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
#alpha = 0.05
#if p > 0.05:
#    print('Sample looks Gaussian (Shapiro) (fail to reject H0)')
#else:
#    print('Sample does not look Gaussian (Shapiro) (reject H0)')

#stat, p = normaltest(dx[iz][ty])
#print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
#alpha = 0.05
#if p > 0.05:
#    print('Sample looks Gaussian (DAgostino/Pearson)(fail to reject H0)')
#else:
#    print('Sample does not look Gaussian (DAgostino/Pearson) (reject H0)')  

#result = anderson(dx[iz][ty])
#print('Statistic: %.3f' % result.statistic)
#p = 0
#for i in range(len(result.critical_values)):
#    sl, cv = result.significance_level[i], result.critical_values[i]
#    if result.statistic < result.critical_values[i]:
#        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
#    else:
#        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))  
           
#sys.exit()  
  
def histogram_plot(ix):  
  ax = dx[ix].plot.hist(bins=50, alpha=0.5)
  plt.xlabel('cm')
  if ix:
    ax.set(title=species[(ix-1)])
  else:
    ax.set(title="All species")  
  plt.show()
  plt.close()
=======
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf

# Display correlation between columns
# ref https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3
def plot_correlation(ix):
# ref https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe 
<<<<<<< HEAD

# ref https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
  fig = plt.figure(figsize=(8,6))
=======
  fig = plt.figure()
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
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
<<<<<<< HEAD
    atitle = 'Pearson correlation for Sepal/Petal length - ' + species[(ix-1)]
  else:
    atitle = 'Pearson correlation for Sepal/Petal length - ' + sl
=======
    atitle = 'Pearson correlation for Sepal/Petal length for ' + species[(ix-1)]
  else:
    atitle = 'Pearson correlation for Sepal/Petal length for ' + sl
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
  ax.set(title=atitle)
  plt.show()
  
def print_data():
  print("\n",df,"\n")
<<<<<<< HEAD
  print("(units - cm)")
=======
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
  
# ref https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas/17071908
# print(df.loc[df['species'] == species[0]])

# ref https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas/17071908
# print(df.loc[(df[columnNames[0]] >= 4.0) & (df[columnNames[0]] <= 5.0)])

# ref https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# print(df.groupby(columnNames[4]).size())

def show_species(ix):
  print("\n",dx[(ix+1)],"\n")
<<<<<<< HEAD
  print("(units - cm)")
  
# ref https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
  
def plot_scatter_ss():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[i].plot(x="sepal_length", y="sepal_width" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Sepal comparison ',xlabel='sepal-length (cm)',ylabel='sepal-width (cm)')
  ax.legend()
  plt.show()
  plt.close()  

def plot_scatter_pp():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[i].plot(x="petal_length", y="petal_width" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Petal comparison ',xlabel='petal-length (cm)',ylabel='petal-width (cm)')
  ax.legend()
  plt.show()
  plt.close()    
  
def plot_scatter_psl():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[i].plot(x="petal_length", y="sepal_length" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Petal-Sepal length comparison ',xlabel='petal-length (cm)',ylabel='sepal-length (cm)')
  ax.legend()
  plt.show()
  plt.close() 
  
def plot_scatter_psw():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[i].plot(x="petal_width", y="sepal_width" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Petal-Sepal width comparison ',xlabel='petal-width (cm)',ylabel='sepal-width (cm)')
  ax.legend()
  plt.show()
  plt.close()         
      
=======
  
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
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
  
# ref https://stackoverflow.com/questions/49970309/how-do-i-calculate-the-mean-of-each-species-of-the-iris-data-set-in-python
# ref https://www.tutorialspoint.com/python_pandas/python_pandas_descriptive_statistics.htm
def print_stats():
<<<<<<< HEAD
  print(bld + "\nStats for ",sl," (units - cm)" + blx)
  print(statall)
  print(" ")
  ot = str(statall)
  
def print_statsx(ix):
  print(bld + "\nStats for ",species[ix]," (units - cm)" + blx)
=======
  print("\nStats for ",sl)
  print(statall)
  print(" ")
  
def print_statsx(ix):
  print("\nStats for ",species[ix])
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
  print(stat[ix])
  print(" ")
  
def close_all():
  master.quit()
  master.destroy()
<<<<<<< HEAD
  
=======
        
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
master = Tk()

Button(master, text='Show all data', command=print_data).grid(row=0, column=0, sticky=W, pady=4)
for i in range(3):
  Button(master, text='Data for ' + species[i], command=partial(show_species,i)).grid(row=0, column=(i+1), sticky=W, pady=4)
<<<<<<< HEAD
  
=======

>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf
Button(master, text='Show overall stats', command=print_stats).grid(row=1, column=0, sticky=W, pady=4)
for i in range(3):
  Button(master, text='Stats for ' + species[i], command= partial(print_statsx,i)).grid(row=1,column=(i+1),sticky=W,pady=4)
  
Button(master, text='Overall correlations', command= partial(plot_correlation,0)).grid(row=2, column=0, sticky=W, pady=4)
for i in range(1,4):
  Button(master, text='Correlations for ' + species[(i-1)], command= partial(plot_correlation,i)).grid(row=2, column=(i), sticky=W, pady=4)
<<<<<<< HEAD

Button(master, text='Overall frequency plot', command= partial(histogram_plot,0)).grid(row=3, column=0, sticky=W, pady=4)
for i in range(1,4):
  Button(master, text='Frequency plot for ' + species[(i-1)], command= partial(histogram_plot,i)).grid(row=3, column=(i), sticky=W, pady=4)
         
Button(master, text='Petal Len-Width scatter'   , command=plot_scatter_pp).grid (row=4, column=0, sticky=W, pady=4)
Button(master, text='Sepal Len-Width scatter'   , command=plot_scatter_ss).grid (row=4, column=1, sticky=W, pady=4)
Button(master, text='Petal-Sepal Length scatter', command=plot_scatter_psl).grid(row=4, column=2, sticky=W, pady=4)
Button(master, text='Petal-Sepal Width scatter' , command=plot_scatter_psw).grid(row=4, column=3, sticky=W, pady=4)
# ref https://stackoverflow.com/questions/6920302/how-to-pass-arguments-to-a-button-command-in-tkinter/22290388

Button(master, text='Show photos', command=show_photos).grid(row=5, column=0, sticky=W, pady=4)
Button(master, text='Help', command=helptext).grid(row=6, column=0, sticky=W, pady=4)
Button(master, text='Quit', command=close_all).grid(row=7, column=0, sticky=W, pady=4)
=======
  
Button(master, text='Scatter graphs', command=plot_scatter).grid(row=3, column=0, sticky=W, pady=4)
# ref https://stackoverflow.com/questions/6920302/how-to-pass-arguments-to-a-button-command-in-tkinter/22290388

Button(master, text='Help', command=helptext).grid(row=4, column=0, sticky=W, pady=4)
Button(master, text='Quit', command=close_all).grid(row=5, column=0, sticky=W, pady=4)
>>>>>>> f347d7dbd175ace7f8f3f3bdbfe0f4e3f46d7cbf

mainloop()