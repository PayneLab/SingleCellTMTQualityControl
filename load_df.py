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

def by_sample(ms3data, technical_replicates):
    #separates the data from readin into the samples
    msSamples = {}
    for sample in technical_replicates:
        reps = {}
        for rep in technical_replicates[sample]:
            reps[ms3data.iloc[:,rep].name] = ms3data.iloc[:,rep]
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
    plt.xscale('log')
    plt.title(title)
    
    for key in channels:
        column = data[key]
        column = np.sort(column.values)
        
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
def ROC_plot(msdata, neg_col_name, technical_replicates, rep_name, as_fraction=False):
    #Generates the points for the curve showing
    #    y-axis: how many sample points are included
    #    x-axis: how many points from the negative control are
    #    as the threshold changes.
    #    See the exaggerated curve above for further clarification.
    #
    #    as_fraction:
    #      True: generates the curves scaled to total number, as decimal
    #      False: generates curves in terms of absolute number of proteins
    #
    #    returns a dictionary of points.
    #    must then be plotted by plt.plot(points.values(), points.keys())
    
    samples = by_sample(msdata, technical_replicates)
    neg_cont = msdata.loc[:,neg_col_name]
    neg_cont = np.array(neg_cont)

    sample = np.array(samples[rep_name].values.flatten())

    all_data = np.concatenate((neg_cont, sample))
    all_data = np.unique(all_data)
    all_data.sort()
    all_data = all_data[::-1]
    
    points = {}
    total = len(all_data)
    for t in all_data:
        x = len([i for i in neg_cont if i > t])
        if as_fraction: x=x / len((neg_cont))
        y = len([i for i in sample if i > t])
        if as_fraction: y=y / len(skipZero(sample))
        points[y] = x
            
    return points

def ROC_all(data, neg_col, cols=list(range(0,10)), boost=None, as_fraction=False, labels=None, title="All Channels"):
    #Calculates and graphs the ROC-like curve for all columns in range.
    #    specifying the boost draws it first, coloring it blue
    #    as_fraction:
    #      True: generates the curves scaled to total number, as decimal
    #      False: generates curves in terms of absolute number of proteins
    plt.xlabel("Control Proteins")
    plt.ylabel("Sample Proteins")
    
    if boost==None: boost_index = None
    else: boost_index = data.columns.get_loc(boost)
    
    if boost!=None:
        p = ROC_plot(data, neg_col, {'a':[boost_index]}, 'a', as_fraction=as_fraction)
        if labels:
            plt.plot(p.values(), p.keys(), label=labels[boost])
        else:
            plt.plot(p.values(), p.keys())
    for i in cols:
        if i != data.columns.get_loc(neg_col) and i != boost_index:
            p = ROC_plot(data, neg_col, {'a':[i]}, 'a', as_fraction=as_fraction)       
            if labels:
                label = labels[(data.columns.values)[i]]
                plt.plot(p.values(), p.keys(), label=label)
            else:
                plt.plot(p.values(), p.keys())
    if labels: plt.legend(loc='lower right')

### Neg to sample ratios - used in figureD
def get_ratios(blank, sample):
    ratios = []
    for protein in sample.index.values:
        sample_value = sample[protein]
        blank_value = blank[protein]
        if sample_value!=0: ratios.append(blank_value/sample_value)
        
    return ratios

def pdc_plot(data):
    pdc_points = []
    data.sort()
    for i in data:
        x = len([ y for y in data if y <= i])/len(data)*100
        pdc_points.append(x)
    return ([data,pdc_points])

def hist_ratios(data, channels, blank,title="Noise Ratios", details=True, log_scale=True, show_zeros=False, pdc=True):
    
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
    