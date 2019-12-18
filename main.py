import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import os
import model_functions as fns
import time
import sys

"""
Load in real data to get sampling frequency
"""

chippies = pd.read_csv("C:/Users/abiga\Box "
                       "Sync\Abigail_Nicole\ChippiesProject"
                       "\FinalDataCompilation"
                       "\AnimalBehaviour_SupplementalDataTable2_addedMid.csv")

yrs_freq = chippies.RecordingYear.value_counts().sort_index().reindex(range(
    1918, 2018, 1)).fillna(0)

sample_freq = yrs_freq.to_numpy(dtype='int')


"""
Cultural Transmission Model
"""
# runs = {'Neutral_0error_40mortality_100iters_100dim':
#             ['neutral', None, None]}
# runs = {'Neutral_0.01error_40mortality_100iters_100dim':
#             ['neutral', 0.0001, None]}
# runs = {'Neutral_0.05error_40mortality_100iters_100dim':
#             ['neutral', 0.0005, None]}
# runs = {'Neutral_0.25error_40mortality_100iters_100dim':
#             ['neutral', 0.0025, None]}
# runs = {'Neutral_0.5error_40mortality_100iters_100dim':
#             ['neutral', 0.005, None]}
# runs = {'Neutral_0.1error_40mortality_100iters_100dim':
#             ['neutral', 0.001, None]}
# runs = {'Neutral_1error_40mortality_100iters_100dim':
#             ['neutral', 0.01, None]}
# runs = {'Conformity_0error_40mortality_100iters_100dim':
#         ['conformity', None, None]}
# runs = {'Conformity_0.5error_40mortality_100iters_100dim':
#         ['conformity', 0.005, None]}
# runs = {'Conformity_0.1error_40mortality_100iters_100dim':
#         ['conformity', 0.001, None]}
# runs = {'Conformity_1error_40mortality_100iters_100dim':
#             ['conformity', 0.01, None]}


home_dir = 'C:/Users/abiga\Box ' \
           'Sync\Abigail_Nicole\ChippiesSyllableModel' \
           '/RealYearlySamplingFreq'
runs = {}
model_type = 'conformity'
direction = None

iterations = 100
low_prop = 0
high_prop = 100
dim = 500
mortality_rate = 0.4

for p in np.arange(0, 0.11, 0.01):
    file_name = model_type + '_' \
                + str(p) + 'error_' \
                + str(int(mortality_rate*100)) + 'mortality_' \
                + str(iterations) + 'iters_' \
                + str(dim) + 'dim'
    runs.update({file_name: [model_type, p/100, direction]})


for run, params in runs.items():
    print(run)
    start_time = time.time()
    path = home_dir + '/' + str(dim) + 'DimMatrix/' + run + '/'
    os.mkdir(path)
    os.chdir(path)

    bird_matrix, init_counts = fns.initiate(low_prop, high_prop, dim,
                                            iterations)
    total_territories = dim**2
    num_deaths = int(mortality_rate*total_territories)

    # initialize figure for video frames
    want_to_save = False
    video_name = run + '.mp4'
    dpi = 100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    frame = ax.imshow(bird_matrix, cmap='gray')
    # cbar = fig.colorbar(frame)
    fig.set_size_inches([5, 5])
    frames = [[frame]]

    prop_counts = init_counts
    bird_counts = init_counts
    sample_counts = np.zeros((high_prop, iterations + 1), dtype='int')

    # don't need to do if we don't care about sampling before 1950
    # samples = fns.sample_birds(bird_matrix, sample_freq[0])
    # for i in samples:
    #     sample_counts[i, 0] += 1

    for timestep in range(iterations):
        print('timestep', timestep)
        # some percent of birds die
        open_territories = fns.locate_dead_birds(num_loc=num_deaths,
                                                 matrix_dim=dim)
        new_props = []
        for bird in open_territories:
            # get new sylls for birds that will now occupy empty territories
            neighbor_sylls = fns.get_nearby_syllables(bird_matrix, bird[0],
                                                      bird[1], d=1)

            # print(neighbor_sylls)
            new_props.append(fns.get_learned_syll(neighbor_sylls,
                                                  rule=params[0],
                                                  error_rate=params[1],
                                                  direction=params[2]))

        # before adding new birds, shift syllable values (bird matrix and prop count) to make room for invented syllables
        unique_props = [list(x) for x in set(tuple(x) for x in new_props)]
        # print(new_props)
        # print('unique props', unique_props)
        for new in unique_props:
            # print('item in unique props', new)
            if new[1] == 'no':
                pass
            else:
                # increase all syllables in bird_matrix by 1 if greater than or equal to invented syllable
                bird_matrix[bird_matrix >= new[0]] = bird_matrix[bird_matrix >= new[0]] + 1
                # must also increase the invented syllables by 1 if greater than the invented syllable
                # this has to be done on both the full and unique lists
                for item in unique_props:
                    if item[0] > new[0]:
                        item[0] = item[0] + 1
                for item in new_props:
                    if item[0] > new[0]:
                        item[0] = item[0] + 1

                # add row of zeros into the count matrix for each invented syllable (row # is the syllable #)
                prop_counts = np.insert(prop_counts, new[0], np.zeros(prop_counts.shape[1]), axis=0)
                bird_counts = np.insert(bird_counts, new[0], np.zeros(bird_counts.shape[1]), axis=0)
                sample_counts = np.insert(sample_counts, new[0], np.zeros(sample_counts.shape[1]), axis=0)

                # print('timestep', timestep, 'num sylls', len(prop_counts))

        # add new birds to the open territories (where old birds died)
        for bird, prop in zip(open_territories, [item[0] for item in new_props]):
            bird_matrix[bird[0], bird[1]] = prop
            bird_counts[prop, timestep + 1] += 1  # count number of new birds with a property

        # count syllables
        prop_counts = fns.count_type(bird_matrix, prop_counts, timestep + 1)

        # sample the birds at this timestep (don't need to sample before 1950)
        if timestep > 31:
            samples = fns.sample_birds(bird_matrix, sample_freq[timestep])
            for i in samples:
                sample_counts[i, timestep + 1] += 1

        new_frame = ax.imshow(bird_matrix, cmap='gray')

        frames.append([new_frame])

    video = animation.ArtistAnimation(fig, frames, interval=100, blit=False,
                                      repeat_delay=1000)
    if want_to_save:
        video.save(video_name)

    plt.close()

    np.savetxt('init_counts.csv', init_counts, delimiter=",")
    np.savetxt('prop_counts.csv', prop_counts, delimiter=",")
    np.savetxt('sample_counts.csv', sample_counts, delimiter=",")
    np.savetxt('bird_counts.csv', bird_counts, delimiter=",")
    print("--- %s seconds ---" % (time.time() - start_time))

    del(init_counts, prop_counts, sample_counts, bird_counts)

