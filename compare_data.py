from __future__ import print_function
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns; sns.set()
from scipy import stats
import numpy as np
import csv
from scipy.stats import ranksums
from matplotlib.ticker import FuncFormatter
import os
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

r_stats = importr('stats')


"""
Set for plotting
"""
c_all = ['#54278f',
         '#5a7d7c',
         '#5fad56',
         '#f2c14e']
c_neutral = ['#54278f',
             '#acbebd',
             '#7a9796',
             '#5a7d7c',
             '#486463',
             '#364b4a'
             ]
c_conformity = ['#54278f',
                '#afd6aa',
                '#7ebd77',
                '#5fad56',
                '#4c8a44',
                '#396733'
                ]

c_directional = ['#54278f',
                 '#f8e0a6',
                 '#f4cd71',
                 '#f2c14e',
                 '#c19a3e',
                 '#91732e'
                 ]


save = True
dispRate = '0.1dispRate'
file_name = 'Neutral_degError_' + dispRate

load_errors = [0.0001, 0.001, 0.01, 0.1, 1.0]
# load_errors = [0.001, 0.01, 0.1, 1.0]
# load_errors = [0.01]
# load_errors = [0.05]

get_models_w = 'neutral'
c = c_neutral

save_path = "C:/Users/abiga/Box " \
            "Sync/Abigail_Nicole/ChippiesSyllableModel" \
            "/RealYearlySamplingFreq/DispersalDist11"

"""
Load in real song data
"""

# song meta data
data_path1 = 'C:/Users/abiga\Box ' \
             'Sync\Abigail_Nicole\ChippiesProject\FinalDataCompilation' \
             '/AnimalBehaviour_SupplementalDataTable2_addedMid.csv'
log_song_data = pd.read_csv(data_path1, header=0, index_col=None)
log_song_data_unique = log_song_data.loc[log_song_data[
    'ComparedStatus'].isin(['unique', 'use'])].copy().reset_index(drop=True)

# get rid of unnecessary metadata and the song stat variables
col_to_skip = ['FromDatabase', 'ComparedStatus', 'RecordingDay',
               'RecordingMonth'] + \
              list(log_song_data_unique.columns[10:26].values)
song_info = log_song_data_unique.drop(col_to_skip, axis=1)

"""
Load in real syllable cluster data
"""
# syllable clusters
data_path2 = "C:/Users/abiga\Box " \
             "Sync\Abigail_Nicole\ChippiesProject" \
             "\StatsOfFinalData_withReChipperReExported\SyllableAnalysis" \
             "\SyllableClusters_820UniqueUse.csv"
cluster_data = pd.read_csv(data_path2, header=0, index_col=None)

col_to_skip2 = ['SyllableNumber', 'ClusterNo']
cluster_data = cluster_data.drop(col_to_skip2, axis=1)
cluster_data['ClusterNoAdjusted'] = cluster_data[
    'ClusterNoAdjusted'].astype(int)

# combine tables using CatalogNo
combined_table = song_info.merge(cluster_data, how='inner', on='CatalogNo')
combined_table = combined_table.drop_duplicates(['CatalogNo',
                                                 'ClusterNoAdjusted'],
                                                keep='first')
combined_table = combined_table.drop(['FileName'], axis=1)


## downsample by latitude and longitude
# combined_table = combined_table.groupby(
#     ['Latitude', 'Longitude']).apply(
#     lambda x: x.sample(1, random_state=42)).reset_index(drop=True)


"""
Load in model data
"""

path_to_model = 'C:/Users/abiga\Box ' \
                'Sync\Abigail_Nicole\ChippiesSyllableModel' \
                '/RealYearlySamplingFreq/DispersalDist11/'

model_details = "1000iters_500dim_500initSylls_40mortRate_" + dispRate

model_lifetimes = {}
model_counts = {}

for i in ['neutral_', 'conformity_', 'directional_']:
    for j in load_errors:
        key_ij = i + str(j) + "err_" + model_details
        try:
            os.chdir(path_to_model + key_ij)
            model_lifetimes[key_ij] = pd.read_csv('sampled_lifetimes.csv',
                                                  dtype='int32', delimiter=',',
                                                  header=None).to_numpy().transpose()[
                0]
            model_counts[key_ij] = \
            pd.read_csv('sampled_birds_with_syllables.csv',
                        dtype='int32', delimiter=',',
                        header=None).to_numpy().transpose()[0]
        except:
            pass


"""
Make an output table for supplement that give the total number of recordings 
in each syllable cluster, the earliest and latest observation of such syllable 
and number of such syllable recorded in the east, west, mid, south. 
"""

# get total number of recordings for each syllable cluster
cluster_num_rec = combined_table.groupby('ClusterNoAdjusted').size().reset_index(
    name='NumberOfRecordings').sort_values('ClusterNoAdjusted').set_index('ClusterNoAdjusted')

cluster_category = combined_table.drop_duplicates('ClusterNoAdjusted')[['ClusterNoAdjusted',
                                                                        'Category']].sort_values(
    'ClusterNoAdjusted').set_index('ClusterNoAdjusted')

earliest_latest_rec = combined_table.assign(EarliestYear=combined_table['RecordingYear'].abs(), LatestYear=combined_table[
    'RecordingYear'].abs()).groupby('ClusterNoAdjusted').agg({'EarliestYear': 'min', 'LatestYear': 'max'})
earliest_latest_rec = earliest_latest_rec.fillna(0).astype(int)

summary_table = pd.concat([cluster_num_rec, earliest_latest_rec, cluster_category], axis=1)
summary_table = summary_table.reindex_axis(['NumberOfRecordings', 'EarliestYear', 'LatestYear', 'Category'], axis=1)
# summary_table.to_csv('C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesProject\StatsOfFinalData_withReChipperReExported'
#                      '/SyllableAnalysis/SyllableClusterSummaryTable.csv')


"""
Plot 3: long-lived lifespans
"""
######PLOT 3: common long-lived lifespans
# use this information to create a histogram of the lifespan of the syllable clusters (with hues for quartiles of
# most to least prevalent syllables)
my_dpi = 96
sns.set(style='white')
sns.set_style('ticks')
sns.set_context({"figure.figsize": (20, 7)})
plt.figure(3)

summary_table_yr = summary_table[summary_table['EarliestYear'] != 0]
summary_table_yr = summary_table_yr.assign(Lifespan=(summary_table_yr['LatestYear'] - summary_table_yr[
    'EarliestYear'] + 1))

# print(pd.qcut(summary_table_yr['NumberOfRecordings'], q=5, duplicates='drop'))
summary_table_yr['quantile'] = pd.qcut(summary_table_yr['NumberOfRecordings'], 5, duplicates='drop', labels=False)

lifespan_quantile = summary_table_yr.groupby(['quantile', 'Lifespan']).size().reset_index(name='count').pivot(
    columns='quantile', index='Lifespan')

# make index not skip missing lifespans, makes it so lifespans can be used
# on y-axis
new_index = list(range(min(lifespan_quantile.index), max(lifespan_quantile.index)+1))
lifespan_quantile = lifespan_quantile.reindex(new_index)

# # plot and save figure
# ax = lifespan_quantile.plot(kind='bar', stacked=True, grid=None, width=1, fontsize=10, edgecolor='black', rot=0,
#                             color=['#cbc9e2', '#9e9ac8', '#756bb1', '#54278f'])# sns.color_palette("PRGn", 10))

print('plot 3 data')

lifespans = lifespan_quantile[:].sum(axis='columns')
lifespans = lifespans.to_frame()
lifespans.columns = ['counts']
lifespans['PercentOfTypes'] = lifespans['counts']/len(summary_table_yr)

plot_3_df = pd.DataFrame(columns=['model', 'overlap', 'p-value'])
for key in model_lifetimes:
    if str(get_models_w) in key:
        sample_lifetimes = model_lifetimes[key]

        #add model data
        num_syll_types = np.sum(sample_lifetimes > 0)
        bin_count_syll_types = np.bincount(sample_lifetimes)
        y = bin_count_syll_types / num_syll_types

        print(key)

        # overlap of percent of types
        observed = lifespans['PercentOfTypes'].to_numpy()
        expected = np.pad(y[1:], (0, len(observed)-len(y[1:])), 'constant')
        print('overlap ', np.sum(np.minimum(observed, expected)))

        # fishers test on counts
        observed_c = lifespans['counts'].fillna(0).to_numpy(dtype=int)
        expected_c = np.pad(bin_count_syll_types[1:], (0, len(observed)-len(
            bin_count_syll_types[1:])), 'constant')
        fisher_p = r_stats.fisher_test(observed_c, expected_c,
                                       workspace=20000000)
        print('p-value: {}'.format(fisher_p[0][0]))

        plot_3_df = plot_3_df.append({'model': key,
                                      'overlap': str(np.sum(np.minimum(
                                         observed, expected))),
                                      'p-value': str(fisher_p[0][0])},
                                     ignore_index=True)
        # for some reason it skips the first value in y, this is okay since
        # we don't want the count of 0 birds singing syllables anyways
        lifespans[key] = pd.Series(y)

# plot and save figure
ax = lifespans.plot(kind='bar',
                    grid=None,
                    use_index=True,
                    y=lifespans.columns[1:],
                    width=0.9,
                    fontsize=10,
                    linewidth=0,
                    rot=0,
                    color=c)

plt.ylabel('Percent of Syllable Types')

plt.tight_layout()
if save:
    plt.savefig(save_path +
                "/Hist_LifespansVsPercentTypes/" +
                file_name +
                '.pdf',
                type='pdf', bbox_inches='tight',
                transparent=True)
    plot_3_df.to_csv(save_path + "/Hist_LifespansVsPercentTypes/" +
                     file_name + '.csv', index=False)
plt.show()
plt.close()

"""
Plot 4: bird frequency spectrum
"""

####PLOT 4: number of recordings vs. percent syllable types
# now make histogram of number of syllable types vs number of birds (aka number of recordings_ with each type).
my_dpi = 96
sns.set(style='white')
sns.set_style('ticks')
sns.set_context({"figure.figsize": (20, 7)})
plt.figure(4)

# get how many syllable types (counts) were
# recorded X number of times (NumberOfRecordings)
numSyllablesWithNumRecordings = summary_table.groupby('NumberOfRecordings').size().reset_index(
    name='counts').reset_index(drop=True)
# divide counts (number of syllable types) by total number of syllable types
numSyllablesWithNumRecordings['PercentOfTypes'] = numSyllablesWithNumRecordings['counts']/len(
    summary_table)

# just for exploring the data (not plotting)
above_thresh = numSyllablesWithNumRecordings[numSyllablesWithNumRecordings['NumberOfRecordings'] >= 20]
percent_numRec = (above_thresh['NumberOfRecordings'].astype(float)*above_thresh['counts']).sum()/(
    numSyllablesWithNumRecordings['NumberOfRecordings']*numSyllablesWithNumRecordings['counts']).sum()
percent_syllTypes = above_thresh['PercentOfTypes'].sum()
# print('percent num rec', percent_numRec)
# print('percent syll types', percent_syllTypes)

numSyllablesWithNumRecordings.set_index('NumberOfRecordings', inplace=True)
# add missing indexes filling in counts and PercentOfTypes w/ NaN
new_index2 = list(range(min(numSyllablesWithNumRecordings.index), max(numSyllablesWithNumRecordings.index)+1))
numSyllablesWithNumRecordings = numSyllablesWithNumRecordings.reindex(new_index2)

print('plot 4 data')

# plot_4_df = pd.DataFrame(columns=['model', 'overlap', 'p-value'])
for key in model_counts:
    if str(get_models_w) in key:
        sample_counts = model_counts[key]

        #add model data
        num_syll_types = np.sum(sample_counts > 0)
        bin_count_syll_types = np.bincount(sample_counts)
        # x = np.arange(len(bin_count_syll_types))[1:]
        y = bin_count_syll_types / num_syll_types

        print(key)

        # overlap of percent of types
        observed = numSyllablesWithNumRecordings[
            'PercentOfTypes'].fillna(0).to_numpy()

        if len(observed) > len(y[1:]):
            expected = np.pad(y[1:], (0, len(observed)-len(y[1:])),
                              'constant')
        else:
            observed = np.pad(observed, (0, len(y[1:])-len(observed)),
                              'constant')
            expected = y[1:]

        # print(stats.chisquare(observed, expected))
        print('overlap ', np.sum(np.minimum(observed, expected)))

        # fishers test on counts
        observed_c = numSyllablesWithNumRecordings['counts'].fillna(
            0).to_numpy(dtype=int)
        expected_c = np.pad(bin_count_syll_types[1:], (0, len(observed)-len(
            bin_count_syll_types[1:])), 'constant')
        # fisher_p = r_stats.fisher_test(observed_c, expected_c,
        #                                workspace=20000000,
        #                                simulate_p_value=True, B=4000)
        # print('p-value: {}'.format(fisher_p[0][0]))
        #
        # plot_4_df = plot_4_df.append({'model': key,
        #                               'overlap': str(np.sum(np.minimum(
        #                                  observed, expected))),
        #                               'p-value': str(fisher_p[0][0])},
        #                              ignore_index=True)
        # for some reason it skips the first value in y, this is okay since
        # we don't want the count of 0 birds singing syllables anyways
        numSyllablesWithNumRecordings[key] = pd.Series(y)

# df index 'NumberOfRecordings' is x-axis
# y is columns 'PercentOfTypes_' for real data and models
# column 'counts' is not used
ax = numSyllablesWithNumRecordings.plot(kind='bar',
                                        use_index=True,
                                        y=numSyllablesWithNumRecordings.columns[1:],
                                        grid=None,
                                        rot=0,
                                        width=0.9,
                                        fontsize=10,
                                        linewidth=0,
                                        color=c)

plt.ylabel('Percent of Syllable Types')

plt.tight_layout()
if save:
    plt.savefig(save_path +
                "/Hist_NumRecVsPercentTypes/" +
                file_name + '.pdf',
                type='pdf', bbox_inches='tight',
                transparent=True)
    # plot_4_df.to_csv(save_path + "/Hist_NumRecVsPercentTypes/" +
    #                  file_name + '.csv', index=False)
plt.show()
plt.close()