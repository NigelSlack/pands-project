#---------------------------------------------------------------------------------------------------#
# Program      : iris_dataset.py
# Author       : Nigel Slack
# Language     : python
#
# Function     : Use a GUI to display the data and simple textual and graphical statistical 
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
# 18/03/2019 NS Put scatter plots in separate outputs, with 2 extra ones - petal/sepal len/width 
# 21/03/2019 NS Include photos        
# 23/03/2019 NS Include Normal Fit plots and analysis by probability distribution functions
# 25/03/2019 NS Put output text into GUI boxes
# 27/03/2019 NS Tidy up statistical tests output
# 28/03/2019 NS Remove Median as a separate stats value - already provided by the 'Describe' 50%
#               percentile.
# 28/03/2019 NS Include histogram plot combining characteristics of each species
#               Use variable for numbering Button rows, rather than hard-code
# 03/04/2019 NS Include probability check. Set colours on buttons. Re-order buttons. 
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
from tkinter import *
from scipy.stats import *
from pylab import imread,subplot,imshow,show
from statsmodels.graphics.gofplots import qqplot
from sklearn.neighbors import KernelDensity

saveout=sys.stdout



hlptext = """\n Display data and statistical analyses of the 'well known' Iris data set.
 The data of petal length/width and sepal length/width, provided by a collection of fifty samples each of three species of
 Iris collected by biologist Edgar Anderson, were analysed in a paper in 1936 by statistician Ronald Fisher. Since then this 
 data set has been widely used for testing and developing statistical techniques.

 This program provides an easy to use GUI to examine the data itself, simple statistical features of the data (in text 
 and graphical form), and a probability check of inputs by the user for a given length, species and characteristic being 
 found in a range (< or > the input value) for a random sample. eg if the user entered a length of '4', then ticked the
 boxes for 'Setosa', 'Sepal length' and '<', the output would be the estimated probability that a random sample for the 
 selected species and characteristic would be less than 4cm long.

 Simply click on the buttons, labelled according to function, to display data, or enter a length (in cm) and tick the
 boxes of one species, characteristic and '<' or '>'.

 The stats check applies 3 commonly used models for testing how 'normal' the data are - that is, how well the data fit a 
 classic bell shaped curve - with the bulk of samples clustered around the mean length, tailing off rapidly for lengths 
 significantly different from the mean.
 
 The graphical outputs provide a quick visual assessment to compare and contrast correlations and distributions for  
 each species and characteristic.

 Syntax : python iris_dataset.py [help]"""
 
# ref https://www.academia.edu/21960049/Step_by_step_practical_guide_with_Statistics_from_ANOVA_to_survival_analysis_in_Biological_Sciences
summout = """\n The statistical tests and plots indicate the following  :
 A very clear distinction exists between the petal length and width for Setosa compared to Versicolor and Virginica, with 
 the Setosa lengths being far smaller (by a factor of 3 to 5). The mean petal width of Versicolor is the characteristic that
 shows the greatest percentage difference from that for Virginica, and would be the best characteristic for distinguishing
 the two.
 All three species exhibit different strengths of correlation between characteristics - for Setosa Sepal length/width are
 strongly correlated, for Versicolor it is Petal length/width and Petal/Sepal length, and for Virginica Sepal/Petal width
 shows the strongest correlation.
 The histogram frequency plots demonstrate that for Setosa there is little tailing off for smaller values of petal width.
 The amount of 'spread' (variance) for characteristics differs between Setosa and the other two - for Setosa sepal 
 length/width is most varied, for Versicolor/Virginica it is sepal/petal length.
 The statistical tests reflect the evidence from the histograms, with Setosa petal width being determined by all elements
 of the tests to be non-Gaussian (normal/bell shaped).
 The normal fit plots show that Setosa petal widths demonstrate the greatest level of clumping around discrete values, thus
 deviating the most from the 'normal' line.
 The scatter graphs for all four comparisons indicate that the features for Setosa are quite distinct from the other
 two. For Versicolor/Virginica, petal length/width are slightly greater for Virginica, with sepal length/width being
 highly intermingled.

 Conclusion : Using a combination of the distinctions highlighted above, an inexperienced botanist collecting random 
 samples gathered in the field should easily be able to distinguish between Setosa and the other two, but may need to
 examine features other than petal/sepal length/width to distinguish with confidence between Versicolor/Virginica."""

# ref https://stackoverflow.com/questions/26629695/how-to-display-content-of-pandas-data-frame-in-tkinter-gui-window  
def helptext():
  root = Tk() 
  t1 = Text(root, height=24, width=125) 
  t1.pack() 
  class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass   
  sys.stdout = PrintToT1() 
  print(hlptext)
  sys.stdout=saveout
  mainloop()   
  
def summary():
  root = Tk() 
  t1 = Text(root, height=24, width=125) 
  t1.pack() 
  class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass   
  sys.stdout = PrintToT1() 
  print(summout)
  sys.stdout=saveout
  mainloop()      
                  
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

# Add variance to the stats provided by describe
#ref https://stackoverflow.com/questions/38545828/pandas-describe-by-additional-parameters
def describe(df):
    return pd.concat([df.describe().T,
                      df.var().rename('variance'),
                     ], axis=1).T
                     
def describex(df,ix):
    return pd.concat([df[df[columnNames[4]]==species[ix]].describe().T,
                      df[df[columnNames[4]]==species[ix]].var().rename('variance'),                      
                     ], axis=1).T                     
                     
pd.set_option('display.max_rows',len(df.index))
# ref https://stackoverflow.com/questions/49188960/how-to-show-all-of-columns-name-on-pandas-dataframe
pd.set_option('display.max_columns', None)
statall=describe(df)
sl = str(df['species'].unique()).strip('[').strip(']')

columnNames = list(df.head(0)) 
species = (df['species'].unique())
co = ['r','b','g']


# Turn bold on/off
# ref https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
# blx = '\033[0m'
# bld = '\033[1m'
dx,stat =([],[])
dx.append(df)

#ref https://stackoverflow.com/questions/52350313/python-for-loop-create-a-list
for i in range (3):
  dx.append(df[df[columnNames[4]]==species[i]])
  stat.append(describex(df[df[columnNames[4]]==species[i]],i))
  

means=[]
stds=[]
# ref https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas
for i in range(4):
  mn=list(df.groupby(columnNames[4])[columnNames[i]].mean())
  st=list(df.groupby(columnNames[4])[columnNames[i]].std())
  for i2 in range(3):
    means.append(mn[i2])
    stds.append(st[i2])  
  
def normal_fit(ix):
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,5.5))
  ax= axes.flatten()  
  for i1 in range(4):
# ref https://thispointer.com/select-rows-columns-by-name-or-index-in-dataframe-using-loc-iloc-python-pandas/  
    cD = dx[ix].loc[ : , columnNames[i1]] 
# ref https://stackoverflow.com/questions/52813683/multiple-qq-plots-in-one-figure    
    qqplot(cD, line='s',ax = ax[i1])
    if ix:
      ax[i1].set_title(species[(ix-1)] + " - " + columnNames[i1])
    else:
      ax[i1].set_title("All species - " + columnNames[i1])
# Use tight_layout so the title and axes labels don't overlap
# ref https://matplotlib.org/users/tight_layout_guide.html
  plt.tight_layout()
  plt.show()
    
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
  

def histogram_plot2(i): 
# ref http://justinbois.github.io/bootcamp/2017/lessons/l21_intro_to_matplotlib.html
  cx=[]
  fig, ax = plt.subplots(1, 1,figsize=(8, 5))
  for ix in range(3):
    cx.append(columnNames[i] + "." + species[ix])
# ref https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html    
    ax.hist(dx[(ix+1)].iloc[0:50,i], bins=50, alpha=0.5,color=co[ix])
  
  ax.legend((cx[0], cx[1], cx[2]), loc='upper right')
  ax.set(title="Comparison " + columnNames[i])
 
  plt.xlabel("cm",fontsize=15)
  plt.ylabel("Frequency",fontsize=15)
  plt.show()
  plt.close()
          
def histogram_plot(ix):  
  ax = dx[ix].plot.hist(bins=100, alpha=0.5)
  plt.xlabel('cm')
  if ix:
    ax.set(title=species[(ix-1)])
  else:
    ax.set(title="All species")  
  plt.show()
  plt.close()

# Display correlation between columns
# ref https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3
def plot_correlation(ix):
# ref https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe 

# ref https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
  fig = plt.figure(figsize=(8,6))
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
    atitle = 'Pearson correlation for Sepal/Petal length - ' + species[(ix-1)]
  else:
    atitle = 'Pearson correlation for Sepal/Petal length - ' + sl
  ax.set(title=atitle)
  plt.show()
    
# ref https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
  
def plot_scatter_ss():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[(i+1)].plot(x="sepal_length", y="sepal_width" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Sepal comparison ',xlabel='sepal-length (cm)',ylabel='sepal-width (cm)')
  ax.legend()
  plt.show()
  plt.close()  

def plot_scatter_pp():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[(i+1)].plot(x="petal_length", y="petal_width" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Petal comparison ',xlabel='petal-length (cm)',ylabel='petal-width (cm)')
  ax.legend()
  plt.show()
  plt.close()    
  
def plot_scatter_psl():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[(i+1)].plot(x="petal_length", y="sepal_length" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Petal-Sepal length comparison ',xlabel='petal-length (cm)',ylabel='sepal-length (cm)')
  ax.legend()
  plt.show()
  plt.close() 
  
def plot_scatter_psw():
  fig,ax=plt.subplots(1,1,figsize=(7,5))
  for i in range (3):
    dx[(i+1)].plot(x="petal_width", y="sepal_width" , kind="scatter",ax=ax,label=species[i],color=co[i])    
  ax.set(title='Petal-Sepal width comparison ',xlabel='petal-width (cm)',ylabel='sepal-width (cm)')
  ax.legend()
  plt.show()
  plt.close()         
  
def close_all():
  master.quit()
  master.destroy()
   

#ref http://python.6.x6.nabble.com/how-to-display-terminal-messages-in-dialog-window-using-tkinter-td1714170.html
# ref https://www.pythoncentral.io/introduction-to-pythons-tkinter/ 
def print_statsx(ix):
  root = Tk() 
  t1 = Text(root) 
  t1.pack() 
  class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass    

  sys.stdout = PrintToT1() 
  print("\nStats for ",species[ix]," (units - cm)\n")
  print(stat[ix])
  print(" ")
  sys.stdout=saveout
  mainloop()

def print_stats():
  root = Tk() 
  t1 = Text(root) 
  t1.pack() 
  class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass    
  sys.stdout = PrintToT1() 
  print("\nStats for ",sl," (units - cm)")
  print(statall)
  print(" ")
  sys.stdout=saveout
  mainloop()      
  
def print_data():
  root = Tk() 
  t1 = Text(root,width=95) 
  t1.pack() 
  class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass    
  sys.stdout = PrintToT1() 
  print("\n",df,"\n")
  print("(units - cm)")
  sys.stdout=saveout
  mainloop()          

def show_species(ix):
  root = Tk() 
  t1 = Text(root) 
  t1.pack() 
  class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass    
  sys.stdout = PrintToT1() 
  print("\n",dx[(ix+1)],"\n")
  print("(units - cm)")
  sys.stdout=saveout
  mainloop()             
                    
def show_stats_checks(ix):
  root = Tk() 
  t1 = Text(root) 
  t1.pack() 
  class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass    
  sys.stdout = PrintToT1() 
  
  pout=[None] * 4
  for i in range(4):
  
# ref https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
      
# ref https://statisticsbyjim.com/hypothesis-testing/interpreting-p-values/

# ref https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

# This means that, in general, we are seeking results with a larger p-value to confirm that our sample 
# was likely drawn from a Gaussian distribution.
# A result above 5% does not mean that the null hypothesis is true. It means that it is very likely true 
# given available evidence. The p-value is not the probability of the data fitting a Gaussian distribution;
# it can be thought of as a value that helps us interpret the statistical test.

# In practice, the Shapiro-Wilk test is believed to be a reliable test of normality, although there is some 
# suggestion that the test may be suitable for smaller samples of data, e.g. thousands of observations or fewer.

# The D’Agostino’s K^2 test calculates summary statistics from the data, namely kurtosis and skewness, to 
# determine if the data distribution departs from the normal distribution, named for Ralph D’Agostino.

# Skew is a quantification of how much a distribution is pushed left or right, a measure of asymmetry in 
# the distribution.
# Kurtosis quantifies how much of the distribution is in the tail. It is a simple and commonly used 
# statistical test for normality.


# Anderson-Darling Test is a statistical test that can be used to evaluate whether a data sample comes from 
# one of among many known data samples, named for Theodore Anderson and Donald Darling.
# It can be used to check whether a data sample is normal. The test is a modified version of a more 
# sophisticated nonparametric goodness-of-fit statistical test called the Kolmogorov-Smirnov test.
# A feature of the Anderson-Darling test is that it returns a list of critical values rather than a single 
# p-value. This can provide the basis for a more thorough interpretation of the result.
# The anderson() SciPy function implements the Anderson-Darling test. It takes as parameters the data sample 
# and the name of the distribution to test it against. By default, the test will check against the Gaussian 
# distribution (dist=’norm’).


# ref http://blog.minitab.com/blog/adventures-in-statistics-2/choosing-between-a-nonparametric-test-and-a-parametric-test
# Reasons to Use Nonparametric Tests
# Reason 1: Your area of study is better represented by the median
# Comparing two skewed distributionsThis is my favorite reason to use a nonparametric test and the one that 
# isn’t mentioned often enough! The fact that you can perform a parametric test with nonnormal data 
# doesn’t imply that the mean is the best measure of the central tendency for your data.
# For example, the center of a skewed distribution, like income, can be better measured by the median 
# where 50% are above the median and 50% are below. If you add a few billionaires to a sample, the 
# mathematical mean increases greatly even though the income for the typical person doesn’t change.
# When your distribution is skewed enough, the mean is strongly affected by changes far out in the 
# distribution’s tail whereas the median continues to more closely reflect the center of the distribution. 
# For these two distributions, a random sample of 100 from each distribution produces means that are 
# significantly different, but medians that are not significantly different.


# ref http://www.scs2.net/next/files/courses/stats/session4.pdf
# Good explanation of hypothesis tests

# ref https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    stat, p = shapiro(dx[ix][columnNames[i]])
    if ix:
      pout[i]=('Statistics for ' + species[(ix-1)] + ' ' + columnNames[i] + ' : ')
    else:
      pout[i]=('Statistics for all species ' + columnNames[i] + ' : ' )  
    if p > 0.05: 
      ga=" - looks Gaussian" 
    else: 
      ga=" - does not look Gaussian"
# ref https://stackoverflow.com/questions/6149006/display-a-float-with-two-decimal-places-in-python
    pout[i]=pout[i]+"\n  Shapiro p-value   " + "{:.2f}".format(p) + ga  

    stat, p = normaltest(dx[ix][columnNames[i]])
    if p > 0.05: 
      ga=" - looks Gaussian" 
    else: 
      ga=" - does not look Gaussian"  
    pout[i]=pout[i]+"\n  Pearson p-value   "+"{:.2f}".format(p)+ga  

    result = anderson(dx[ix][columnNames[i]])
    for i2 in range(len(result.critical_values)):
      sl, cv = result.significance_level[i2], result.critical_values[i2]
      if result.statistic < result.critical_values[i2]:
        if i2 < 2:
          pout[i]=(pout[i]+"\n  Anderson %.3f: %.3f - looks Gaussian" % (sl, cv))
        else:
          pout[i]=(pout[i]+"\n  Anderson  %.3f: %.3f - looks Gaussian" % (sl, cv))
      else:
        if i2 < 2:
          pout[i]=(pout[i]+"\n  Anderson %.3f: %.3f - does not look Gaussian" % (sl, cv))
        else:
          pout[i]=(pout[i]+"\n  Anderson  %.3f: %.3f - does not look Gaussian" % (sl, cv))
       
    pout[i]=(pout[i]+"\n")   
    print(pout[i])     
   
  sys.stdout=saveout
  mainloop()                                         
                                                                   
def check_inputs():

    root = Tk() 
    t1 = Text(root, height=4, width=25) 
    t1.pack() 
    class PrintToT1(object): 
     def write(self, s): 
       t1.insert(END, s) 
     def flush(self):
       pass   
    sys.stdout = PrintToT1() 

 
    i,n1,n2,n3=0,0,0,0 
    spx,chx,cx=[0,0,0],[0,0,0,0],[0,0]
    ok=True
    for v in vars:
      if v.get():
        if i < 3:
          spx[i]=1
          n1+=1
        elif i < 7:
          chx[(i-3)]=1
          n2+=1 
        else:
          cx[(i-7)]=1
          n3+=1 
      i+=1
    
    if (not n1) or (not n2) or (not n3):   
      ok=False
      print("Select one each of species, characteristic and '<' or '>'")
    elif (n1>1) or (n2>1) or (n3>1):
      ok=False
      print("Only select one each of species, characteristic and '<' or '>'")
    else:           
      val = e1.get()
      try:
        val = float(val)
        if val <= 0 or val > 10:
          print("Enter a length in the range 0 < len <= 10")
          ok=False
      except ValueError:
        print ("Invalid length input")
        ok=False

    if ok:
      ind=spx[1]+(spx[2]*2)+(chx[1]*3)+(chx[2]*6)+(chx[3]*9)
      if cx[0]:
        print("Probability =",100*round(norm.cdf(val,means[ind],stds[ind]),2),"%")
      else:  
        print("Probability =",100*round(norm.sf(val,means[ind],stds[ind]),2),"%")
                                                                                              

      
      if cx[0]:
        start = 0
        end = val
      else:
        start = val
        end = 20  
     
      N = 50    # Number of evaluation points 
      step = (end - start) / (N - 1)  # Step size
      x = np.linspace(start, end, N)[:, np.newaxis]  # Generate values in the range
      #kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
      #probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF
      #print(probability)                                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
      sys.stdout=saveout
      mainloop()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                    
master = Tk()
rn=0
vars = []
for i in range(9):
    vars.append(IntVar())
    
# ref https://stackoverflow.com/questions/3949844/how-to-get-the-screen-size-in-tkinter/3949983#3949983
screen_width = master.winfo_screenwidth()    


Button(master, text='Show all data', command=print_data,fg="orange").grid(row=rn, column=0, sticky=W, pady=4)
for i in range(3):
  Button(master, text='Data for ' + species[i], command=partial(show_species,i),fg="orange").grid(row=rn, column=(i+1), sticky=W, pady=4)
rn+=1
    
Button(master, text='Show overall stats', command=print_stats,fg="orange").grid(row=rn, column=0, sticky=W, pady=4)
for i in range(3):
  Button(master, text='Stats for ' + species[i], command= partial(print_statsx,i),fg="orange").grid(row=rn,column=(i+1),sticky=W,pady=4)
rn+=1

Button(master, text='Overall Stats check', command= partial(show_stats_checks,0),fg="orange").grid(row=rn, column=0, sticky=W, pady=4)
for i in range(1,4):
  Button(master, text='Stats check for ' + species[(i-1)], command= partial(show_stats_checks,i),fg="orange").grid(row=rn, column=(i), sticky=W, pady=4)
rn+=1
  
Button(master, text='Overall correlations', command= partial(plot_correlation,0),fg="green").grid(row=rn, column=0, sticky=W, pady=4)
for i in range(1,4):
  Button(master, text='Correlations for ' + species[(i-1)], command= partial(plot_correlation,i),fg="green").grid(row=rn, column=(i), sticky=W, pady=4)
rn+=1

Button(master, text='Overall frequency plot', command= partial(histogram_plot,0),fg="green").grid(row=rn, column=0, sticky=W, pady=4)
for i in range(1,4):
  Button(master, text='Frequency plot for ' + species[(i-1)], command= partial(histogram_plot,i),fg="green").grid(row=rn, column=(i), sticky=W, pady=4)
rn+=1
  
for i in range(0,4):
  Button(master, text='Frequency plot ' + columnNames[i], command= partial(histogram_plot2,i),fg="green").grid(row=rn, column=(i), sticky=W, pady=4)
rn+=1
            
Button(master, text='Overall Normal Fit plot', command= partial(normal_fit,0),fg="green").grid(row=rn, column=0, sticky=W, pady=4)
for i in range(1,4):
  Button(master, text='Normal Fit plot for ' + species[(i-1)], command= partial(normal_fit,i),fg="green").grid(row=rn, column=(i), sticky=W, pady=4)
rn+=1
                  

Button(master, text='Petal Len-Width scatter'   , command=plot_scatter_pp,fg="green").grid (row=rn, column=0, sticky=W, pady=4)
Button(master, text='Sepal Len-Width scatter'   , command=plot_scatter_ss,fg="green").grid (row=rn, column=1, sticky=W, pady=4)
Button(master, text='Petal-Sepal Length scatter', command=plot_scatter_psl,fg="green").grid(row=rn, column=2, sticky=W, pady=4)
Button(master, text='Petal-Sepal Width scatter' , command=plot_scatter_psw,fg="green").grid(row=rn, column=3, sticky=W, pady=4)
rn+=1

# ref https://stackoverflow.com/questions/6920302/how-to-pass-arguments-to-a-button-command-in-tkinter/22290388

Button(master, text='Show photos', command=show_photos,fg="green").grid(row=rn, column=0, sticky=W, pady=4)
rn+=1


Label(master, text="Check probability of value in range :", fg="blue").grid(row=rn, column=0, sticky=W)
rn+=1
Label(master, text="Enter Length; select one species/characteristic and '<' or '>'",fg="blue").grid(row=rn, column=0, sticky=W)
rn+=1

Label(master, text="Enter Length (cm)",width = 22,fg="blue").grid(row=rn, column=0, sticky=W)
e1 = Entry(master,width=3)
e1.grid(row=rn, column=0,sticky=W)
#e1.grid(row=rn, column=1)
rn+=1

Checkbutton(master, text="setosa"    , variable=vars[0],fg="blue").grid(row=rn, column=0, sticky=W)
Checkbutton(master, text="versicolor", variable=vars[1],fg="blue").grid(row=rn, column=1, sticky=W)
Checkbutton(master, text="virginica" , variable=vars[2],fg="blue").grid(row=rn, column=2, sticky=W)
rn+=1  
  
Checkbutton(master, text="sepal_length", variable=vars[3],fg="blue").grid(row=rn, column=0, sticky=W)
Checkbutton(master, text="sepal_width" , variable=vars[4],fg="blue").grid(row=rn, column=1, sticky=W)
Checkbutton(master, text="petal_length", variable=vars[5],fg="blue").grid(row=rn, column=2, sticky=W)
Checkbutton(master, text="petal_width" , variable=vars[6],fg="blue").grid(row=rn, column=3, sticky=W)
rn+=1

Checkbutton(master, text="<", variable=vars[7],fg="blue").grid(row=rn, column=0, sticky=W)
Checkbutton(master, text=">", variable=vars[8],fg="blue").grid(row=rn, column=1, sticky=W)
rn+=1
 
Button(master, text='Check probability', command=check_inputs,fg="blue").grid(row=rn, sticky=W, pady=4)
rn+=1
 
Button(master, text='Summary', command=summary,fg="red").grid(row=rn, column=0, sticky=W, pady=4)
rn+=1
Button(master, text='Help', command=helptext,fg="red").grid(row=rn, column=0, sticky=W, pady=4)
rn+=1
Button(master, text='Quit', command=close_all,fg="red").grid(row=rn, column=0, sticky=W, pady=4)


mainloop()