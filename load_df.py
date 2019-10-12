#!/usr/bin/env python
from statistics import mean
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(fileName):
    inFile = open(fileName, 'r')
    
    line = inFile.readline().strip()
    headings = line.split('\t')
       
    data = {}
    line = inFile.readline()
    while line !="":
        row = line.split('\t')
        vals = row[1:]
        for i in vals: i = float(i)
        data[(row[0])] = vals
        line = inFile.readline()
        
    df = pd.DataFrame.from_dict(data, dtype = float, orient='index',columns = headings[1:])
    return df

def readin_log(fileName):
    df = readin(fileName)
    dfl = (np.log(df)).replace(-np.inf, 0)
    return dfl

def by_sample(data, technical_replicates):
    #separates the data from readin into the samples
    #   data is the dataframe returned by load_df
    #   technical_replicates is a dict arranged as
    #     "Condition name":[0,1,2]#channel numbers
    msSamples = {}
    for sample in technical_replicates:
        reps = {}
        for rep in technical_replicates[sample]:
            reps[data.iloc[:,rep].name] = data.iloc[:,rep]
        msSamples[sample] = pd.DataFrame.from_dict(reps, dtype = float)
    
    return msSamples

def n_thresholds(alist, percents=[95], display=True):
    ###Given a list calculates thresholds.
    #   defaults to calculating the 95% threshold
    #   optionally prints the thresholds calculated
    #   returns a dictionary with the thresholds
    #       including and excluding zeros for each
    
    alist = sorted(alist, reverse=True)
    with_zeros = {}
    for i in percents:
        p = (100.0-float(i))/100.0
        t = float(alist[math.ceil(float(len(alist))*p)])
        with_zeros[i] = t
        
        if display: print("{0}% threshold: {1}".format(i, t))

    if display: print("\nIgnoring Zeros: ")
    alist = [x for x in alist if (x!=0)]
    skip_zeros = {}
    for i in percents:
        p = (100.0-float(i))/100.0
        t = float(alist[math.ceil(float(len(alist))*p)])
        skip_zeros[i] = t
        
        if display: print("{0}% threshold: {1}".format(i, t))
    
    r = {
        'with_zeros':with_zeros,
        'without_zeros':skip_zeros
    }
    
    return r

### Graphed types - as used in figure B
def hist_comp_channels(data, channels,title="Neg Control vs Samples", show_zeros=False):
    #Creates a histogram of selected channels, generally samples and blank.
    #    data is a pandas dataframe as returned by load_df
    #    channels is a dictionary where keys are the column name in data
    #    and the value is the desired label, such as "Cell X"
    #    show_zeros controls whether the zero values are shown. 
    #    Since in protein data this means it was not detected,
    #    zeros are left out here by default, but may be included if desired.
    plt.xscale('log')
    plt.title(title)
    
    for key in channels:
        column = data[key]
        column = np.sort(column.values)
        
        #This scales the histogram to the data.
        minx = np.log10(min([x for x in column if x != 0])) -.5
        maxx = np.log10(max(column)) +.5
        bins = np.logspace(minx, maxx)
        if show_zeros:
            x = [0]
            for i in bins: x.append(i)
            bins = x
        plt.hist(column, alpha = .5, bins=bins, label=channels[key])
        
    plt.legend(loc='upper right')
    plt.xlabel("Intensity Value")
    plt.ylabel("Number of Proteins")

    plt.show()

### ROC graphs - used in figure C
def ROC_plot(msdata, neg_col_name, replicates, rep_name, as_fraction=False, square=False):
    #Generates the points for the curve showing for any threshold
    #    y-axis: how many sample points are included
    #    x-axis: how many points from the negative control
    #
    #    Parameters:
    #    msdata is a dataframe as returned by load_df
    #    neg_col_name is the name of the blank column
    #    replicates is a dict arranged as
    #     "Condition name":[0]#channel numbers
    #    rep_name is the condition name from replicates.
    #    Note that square must be false if the replicate includes more than one.
    #
    #    as_fraction:
    #      True: generates the curves scaled to total number, as decimal
    #      False: generates curves in terms of absolute number of proteins
    #
    #    returns a dictionary of points.
    #    must then be plotted by plt.plot(points.values(), points.keys())
    #    This step is omitted so multiple may be graphed together.
    
    samples = by_sample(msdata, replicates)
    neg_cont = msdata.loc[:,neg_col_name]
    neg_cont = np.array(neg_cont)

    sample = np.array(samples[rep_name].values.flatten())

    all_data = np.concatenate((neg_cont, sample))
    all_data = np.unique(all_data)
    all_data.sort()
    all_data = all_data[::-1]

    # y_max should be greater than x_max, but might not be
    x_max = len([s for s in neg_cont if s != 0])
    y_max = len([s for s in sample if s != 0])
    corner = max(x_max, y_max)
    
    points = {}
    for t in all_data:
        x = len([i for i in neg_cont if i > t])
        if as_fraction and square: x=x / corner
        elif as_fraction: x=x / x_max
        y = len([i for i in sample if i > t])
        if as_fraction and square: y=y / corner
        if as_fraction: y=y / y_max
        points[y] = x
            
    if as_fraction:
        points[1]= 1 #go to corner
    else:
        points[corner]=corner
    return points

def ROC_all(data, neg_col, cols=None, boost=None, as_fraction=False, labels=None, title="All Channels", get_score=False):
    #Calculates and graphs the ROC-like curve for all columns in range.
    #    Parameters:
    #    msdata is a dataframe as returned by load_df
    #    neg_col is the name of the blank column
    #    
    #    cols is a list of column indexes. This should only be specified
    #      if some channels should not be graphed. If it is, specifying
    #      labels as well is recommended.
    #    
    #    boost is the name of the boost channel. Specifying it draws it first,
    #      which colors it blue and lists it first in the legend.
    #    labels is a dictionary of the column names and desired labels
    #    title is passed directly as the plot title.
    #    as_fraction:
    #      True: generates the curves scaled to total number, as decimal
    #      False: generates curves in terms of absolute number of proteins
    #    get_score is a boolean. If true it will calculate, print, 
    #      and return the 'area under the curve' based score by channel
    #      1 is best, anything under .5 is worse than no difference.
    if cols==None:
        cols=cols=list(range(0,len(data.columns))
                       
    plt.xlabel("Control Proteins")
    plt.ylabel("Sample Proteins")
    plt.title(title)
    
    if boost==None: boost_index = None
    else: boost_index = data.columns.get_loc(boost)
    if get_score: areas = {}
    
    if boost!=None:
        p = ROC_plot(data, neg_col, {'a':[boost_index]}, 'a', as_fraction=as_fraction,square=True)
        if labels: label=labels[boost]
        else: label = boost
        plt.plot(p.values(), p.keys(), label=label)
        if get_score:
            area = np.trapz(y=list(p.keys()), x=list(p.values()))
            if not as_fraction: area=area/(max(list(p.keys()))**2)
            areas[labels[boost]]=area
        
    for i in cols:
        if i != data.columns.get_loc(neg_col) and i != boost_index:
            p = ROC_plot(data, neg_col, {'a':[i]}, 'a', as_fraction=as_fraction, square=True)       
            if labels:
                label = labels[(data.columns.values)[i]]
            else:
                label=(data.columns.values)[i]
            plt.plot(p.values(), p.keys(), label=label)
            if get_score:
                area = np.trapz(list(p.keys()),x=list(p.values()))
                if not as_fraction: area=area/(max(list(p.keys()))**2)
                areas[label]=area
    plt.legend(loc='lower right')
    plt.axis('square')
    if get_score:
        print ("Scores (out of 1)")
        for k in areas:
            print ("{0}\t{1:.4f}".format(k,areas[k]))
        return areas

### Neg to sample ratios - used in figureD
def get_ratios(blank, sample):
    #Takes the blank and sample series
    #   and returns for each protein
    #   in the sample the ratio of blank/sample
    ratios = []
    for protein in sample.index.values:
        sample_value = sample[protein]
        blank_value = blank[protein]
        if sample_value!=0: ratios.append(blank_value/sample_value)
    return ratios

def pdc_plot(data):
    #Generate the points for a probability density curve
    #    data is any iterable set of numbers.    
    #    The curve is then plotted with
    #    plt.plot(pdc_points[0], pdc_points[1])
    pdc_points = []
    data.sort()
    for i in data:
        x = len([ y for y in data if y <= i])/len(data)*100
        pdc_points.append(x)
    return ([data,pdc_points])

def hist_ratios(data, channels, blank,title="Noise Ratios", details=True, log_scale=True, show_zeros=False, pdc=True):
    #Plots the noise/signal ratios
    #    data is a dataframe as returned by load_df
    #    channels is a dictionary where keys are the column name in data
    #      and the value is the desired label, such as "Cell X"
    #    details=True prints some extra info, such as average
    #    log_scale controls whether the plot is on a log scale
    #    show_zeros controls whether 0 is graphed
    #    pdc controls whether the probability density curve is shown
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if log_scale:
        plt.xscale('log')
    plt.title(title)
    blank_data = data.loc[:,blank]
    for c in channels:
        sample = data.loc[:,c]#retrives the column
        ratios = get_ratios(blank_data, sample)

        bins = 50
        if log_scale:
            minx = np.log10(min([x for x in ratios if x != 0])) -.25
            maxx = np.log10(max(ratios)) +.5
            bins = np.logspace(minx, maxx)
            if show_zeros:
                x = [0]
                for i in bins: x.append(i)
                bins = x
        ax1.hist(ratios, alpha = .7, bins=bins, label=channels[c])
        
        if details:
            print(channels[c]+':')
            print((len([x for x in ratios if x==0])),'of',len(ratios),'are 0.0')
            print ('Average: {0:.4f}'.format(mean(ratios)))
            threshold95 = n_thresholds(ratios, display=False)['with_zeros'][95]
            print ('95% Threshold: {0:.4f}'.format(threshold95))
            
    pdc_points = pdc_plot(ratios)
    ax2.set_ylim([0,100])
    ax2.plot(pdc_points[0], pdc_points[1], color='orange')
    
    ax1.legend(loc='upper right')
    ax1.set_xlabel("Blank/Sample Ratio")
    ax1.set_ylabel("Number of Proteins")
    ax2.set_ylabel("-- Percent of Proteins")