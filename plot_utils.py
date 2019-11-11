#!/usr/bin/env python

#These functions create plots used for analyzing
#    signal/noise based on a negative control.
#More information may be found at:
#    https://github.com/PayneLab/SingleCellTMTQualityControl

from statistics import mean
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

### Graphed types - as used in figure A
def hist_comp_channels(data, channels,title="Neg Control vs Samples",
        show_zeros=True, save_as=None):
    #Creates a histogram of selected channels, generally samples and blank.
    #    data is a pandas dataframe as returned by load_dataset
    #    channels is a dictionary where keys are the column name in data
    #    and the value is the desired label, such as "Cell X"
    #    show_zeros controls whether the zero values are shown.
    #    Since in protein data this means it was not detected,
    #    zeros may be left out or may be included to demonstrate overlap.

    #To get even bins, we first get the x range
    #that we want to plot.
    min_x_list,max_x_list,zero_heights = [], [], []
    for key in channels:
        column = data[key]
        column = np.sort(column.values)

        #This scales the histogram to the data.
        min_x_list.append(np.log10(min([x for x in column if x != 0])) -.5)
        max_x_list.append(np.log10(max(column)) +.5)

        #These will be the y values for the break tops.
        zero_count = len([i for i in column if i == 0])
        zero_ceil = (math.ceil(zero_count/50))*50
        zero_heights.append(zero_ceil)
    #These will be the extremes we want to plot.
    minx = min(min_x_list)
    maxx = max(max_x_list)
    bins = np.logspace(minx, maxx)
    #These will be the y segments
    zero_heights = list(set(zero_heights))
    zero_heights.sort(reverse=True)#top to bottom

    axs = []
    if show_zeros:
        x = [0]
        for i in bins: x.append(i)
        bins = x

    if show_zeros==True:
        height_ratios = []
        for i in zero_heights: height_ratios.append(1)
        height_ratios.append(7)#height of main, lower plot

        fig, (axs) = plt.subplots(len(zero_heights)+1, 1, sharex=True, gridspec_kw={'height_ratios': height_ratios})

        axs[0].set_title(title)
        for ax, h in zip(axs,zero_heights):
            ax.set_ylim(h-50, h)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none')
            ax.spines['top'].set_visible(False)
            ax.tick_params(labeltop=False)
        axs[0].spines['top'].set_visible(True)
        axs[len(axs)-1].spines['top'].set_visible(False)

    else:
        fig, (ax) = plt.subplots()
        axs.append(ax)
        ax.set_title(title)
    plt.xscale('log')

    #Now we can plot it.
    main_ceils = []
    for key in channels:
        column = data[key]
        column = np.sort(column.values)

        for ax in axs:
            y, x, _ = ax.hist(column, alpha = .5, bins=bins, label=channels[key])
        main_ceils.append((math.ceil(max(y[1:])/10))*10)
    main_ceiling=max(main_ceils)

    ax = axs[len(axs)-1]
    ax.set_ylim(0, main_ceiling)
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.tick_bottom()

    plt.legend(loc='upper right')
    plt.xlabel("Intensity Value")
    plt.ylabel("Number of Proteins")

    if save_as: fig.savefig(save_as, dpi=300)

    plt.show()

def by_sample(data, technical_replicates):
    #separates the data from readin into the samples
    #   data is the dataframe returned by load_data
    #   technical_replicates is a dict arranged as
    #     "Condition name":[0,1,2]#channel numbers
    msSamples = {}
    for sample in technical_replicates:
        reps = {}
        for rep in technical_replicates[sample]:
            reps[data.iloc[:,rep].name] = data.iloc[:,rep]
        msSamples[sample] = pd.DataFrame.from_dict(reps, dtype = float)

    return msSamples

### ROC graphs - used in figure B
def ROC_plot(msdata, neg_col_name, replicates, rep_name,
        as_fraction=False, square=False):
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

    #we combine the two to 'walk through' in the graph
    all_data = np.concatenate((neg_cont, sample))
    all_data = np.unique(all_data)
    all_data.sort()
    all_data = all_data[::-1]

    # y_max should be greater than x_max, but might not be
    x_max = len([s for s in neg_cont if s != 0])
    y_max = len([s for s in sample if s != 0])
    corner = max(x_max, y_max)

    #Here we calculate the points on the graph
    points = {}
    for t in all_data:
        x = len([i for i in neg_cont if i > t])
        if as_fraction and square: x=x / corner
        elif as_fraction: x=x / x_max
        y = len([i for i in sample if i > t])
        if as_fraction and square: y=y / corner
        elif as_fraction: y=y / y_max
        points[y] = x

    if as_fraction:
        points[1]= 1 #go to corner
    else:
        points[corner]=corner
    return points

def ROC_all(data, neg_col, cols=None, boost=None, as_fraction=True,
        labels=None, title="All Channels", get_score=False, save_as=None):
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
        cols=list(range(0,len(data.columns)))

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

    if save_as: plt.savefig(save_as, dpi=300)

    if get_score:
        print ("Scores (out of 1)")
        for k in areas:
            print ("{0}\t{1:.4f}".format(k,areas[k]))
        return areas

### Neg to sample ratios - used in figure C
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

def hist_ratios(data, channels, blank,title="Noise Ratios", details=True,
        log_scale=True, show_zeros=True, pdc=False, cutoff=.2, save_as=False):
    #Plots the noise/signal ratios
    #    data is a dataframe as returned by load_dataset
    #    channels is a dictionary where keys are the column name in data
    #      and the value is the desired label, such as "Cell X"
    #    details=True prints some extra info, such as average
    #    log_scale controls whether the plot is on a log scale
    #    show_zeros controls whether 0 is graphed
            #Options:
    #       True: split y-axis to show zeros
    #       'no_break': show zeros without splitting y-axis
    #       False: No zeros shown
    #    pdc controls whether the probability density curve is shown
    #Returns the percent passing cuttoff

    cutoff_report = {}

    if show_zeros==True:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 4]})
    else: fig, ax1 = plt.subplots()
    if pdc: ax_pdc = ax1.twinx()
    ax1.set_title(title)
    if log_scale:
        plt.xscale('log')
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
            bins=x
        if show_zeros==True:
            ax2.hist(ratios, alpha = .7, bins=bins, label=channels[c])
            y, x, _ = ax1.hist(ratios, alpha = .7, bins=bins, label=channels[c])

            zero_count = y[0]#first x will be zero
            zero_ceil = (math.ceil(zero_count/50))*50
            ax1.set_ylim(zero_ceil-50, zero_ceil)

            main_ceiling = (math.ceil(max(y[1:])/10))*10
            ax2.set_ylim(0, main_ceiling)  # most of the data

            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.xaxis.set_ticks_position('none')
            ax1.tick_params(labeltop=False)
            ax2.xaxis.tick_bottom()

            ax2.set_xlabel("Blank/Sample Ratio")
            ax2.set_ylabel("Number of Proteins")
        else:
            ax1.set_xlabel("Blank/Sample Ratio")
            ax1.set_ylabel("Number of Proteins")
            ax1.hist(ratios, alpha = .7, bins=bins, label=channels[c])

        if pdc:
            pdc_points = pdc_plot(ratios)
            ax_pdc.set_ylim([0,100])
            ax_pdc.plot(pdc_points[0], pdc_points[1], color='orange', label="Probability Density")
            ax_pdc.set_ylabel("-- Percent of Proteins")
        cutoff_percent=(len([x for x in ratios if x < cutoff])/len(ratios))*100
        cutoff_report[c]=cutoff_percent
        ax1.axvline(x=cutoff, linestyle='dashed', label='{0} ({1:.2f}%)'.format(cutoff, cutoff_percent), color='black')
        if show_zeros==True: ax2.axvline(x=cutoff, linestyle='dashed', label='{0} ({1:.2f}%)'.format(cutoff, cutoff_percent), color='black')

        if details:
            print(channels[c]+':')
            print((len([x for x in ratios if x==0])),'of',len(ratios),'are 0.0')
            print ('Average: {0:.4f}'.format(mean(ratios)))
            threshold95 = n_thresholds(ratios, display=False)['with_zeros'][95]
            print ('95% Threshold: {0:.4f} (1 to {1:.1f})'.format(threshold95, 1/threshold95))
            print ("{percent:.2f}% passed a {t} threshold (1 to {T})"
                .format(percent=cutoff_report[c], t=cutoff, T=1/cutoff))

    ax1.legend(loc='upper right')
    if save_as: fig.savefig(save_as, dpi=300)
    return cutoff_report
