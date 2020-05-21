import pandas as pd


def load_recording_data(song_stats=False):
    """
    Load in song data
    """

    # song meta data
    data_path1 = \
        'citizen_science_data' \
        '/AnimalBehaviour_SupplementalDataTable2_addedMid_addedYears.csv'
    log_song_data = pd.read_csv(data_path1, header=0, index_col=None)
    log_song_data_unique = log_song_data.loc[log_song_data[
        'ComparedStatus'].isin(['unique', 'use'])].copy().reset_index(drop=True)

    # get rid of unnecessary metadata and the song stat variables
    if song_stats:
        col_to_skip = ['FromDatabase', 'ComparedStatus', 'RecordingDay',
                       'RecordingMonth', 'RecordingTime', 'Region']
    else:
        col_to_skip = ['FromDatabase', 'ComparedStatus', 'RecordingDay',
                       'RecordingMonth', 'RecordingTime', 'Region'] + \
                      list(log_song_data_unique.columns[10:26].values)
    song_info = log_song_data_unique.drop(col_to_skip, axis=1)

    """
    Load in syllable cluster data
    """
    # syllable clusters
    data_path2 = 'citizen_science_data/SyllableClusters_820UniqueUse.csv'
    cluster_data = pd.read_csv(data_path2, header=0, index_col=None)

    col_to_skip2 = ['SyllableNumber', 'ClusterNo']
    cluster_data = cluster_data.drop(col_to_skip2, axis=1)
    cluster_data['ClusterNoAdjusted'] = cluster_data[
        'ClusterNoAdjusted'].astype(int)

    """
    combine tables using CatalogNo
    """
    combined_table = song_info.merge(cluster_data, how='inner', on='CatalogNo')
    combined_table = combined_table.drop_duplicates(['CatalogNo',
                                                     'ClusterNoAdjusted'],
                                                    keep='first')
    combined_table = combined_table.drop(['FileName'], axis=1)

    return combined_table
