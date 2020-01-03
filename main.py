import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import os
import itertools
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
           '/RealYearlySamplingFreq/Testing4_new'
runs = {}
model_type = 'neutral'
direction = None

iterations = 100
low_prop = 0
high_prop = 100
dim = 500
mortality_rate = 0.4
num_syll_types = high_prop
num_samples = 68  # must be less than iterations, 68 comes from 2017-1950 (dates we have data from)

# get list of all coordinate pairs of matrix
all_coord = list(itertools.product(range(0, dim), range(0, dim)))

# setup runs with various parameters
for p in np.arange(0.05, 0.051, 0.01):
    file_name = model_type + '_' \
                + str(p) + 'error_' \
                + str(int(mortality_rate*100)) + 'mortality_' \
                + str(iterations) + 'iters_' \
                + str(dim) + 'dim'
    runs.update({file_name: [model_type, p/100, direction]})

# iterate through each of the runs, each with unique parameters
for run, params in runs.items():
    print(run)
    start_time = time.time()
    path = home_dir + '/' + str(dim) + 'DimMatrix/' + run + '/'
    os.mkdir(path)
    os.chdir(path)

    # create matrix with each element being a bird
    total_territories = dim ** 2
    num_deaths = int(mortality_rate*total_territories)
    # create empty vector w/ more space than expected for num of sylls created
    vector_size = int(total_territories * iterations * params[1] * \
                  mortality_rate * 10)

    # initialize the first set of birds
    bird_matrix, current_bps, actual_lifetimes, unique_bps, sample_bps, \
        sample_unique_bps, first_sampled, last_sampled = \
        fns.initiate(low_prop, high_prop, dim, iterations, vector_size)

    # find number of birds that have each syllable type, save initial state
    current_bps = fns.count_type(bird_matrix, vector_size)
    bird_counts_t0 = current_bps.copy()

    # initialize figure for video frames
    # want_to_save = False
    # video_name = run + '.mp4'
    # dpi = 100
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    #
    # frame = ax.imshow(bird_matrix, cmap='gray')
    # # cbar = fig.colorbar(frame)
    # fig.set_size_inches([5, 5])
    # frames = [[frame]]

    for timestep in range(iterations):
        print('\ntimestep', timestep)
        # some percent of birds die, find their grid location
        open_territories = fns.locate_dead_birds(ordered_pairs=all_coord,
                                                 num_loc=num_deaths)
        new_props = []  # list of learned syllable types (could be a new type)
        for bird in open_territories:
            # get new sylls for birds that will now occupy empty territories
            neighbor_sylls = fns.get_nearby_syllables(bird_matrix, bird[0],
                                                      bird[1], d=1)

            new_syll, num_syll_types = fns.get_learned_syll(neighbor_sylls,
                                                            num_syll_types,
                                                            rule=params[0],
                                                            error_rate=params[1],
                                                            direction=params[2])
            new_props.append(new_syll)

        # add new birds to the open territories (where old birds died)
        for bird, prop in zip(open_territories, new_props):
            bird_matrix[bird[0], bird[1]] = prop
            # after the first timestep in which we sample, increment the
            # number of birds with a syllable type for each new bird
            if timestep > iterations - num_samples:
                unique_bps[prop] += 1

        # when the timestep is the first year to sample, initiate unique_bps
        if timestep == iterations - num_samples:
            unique_bps = fns.count_type(bird_matrix, vector_size)
            bird_counts_t1950 = unique_bps.copy()

        # sample the birds at this timestep (don't need to sample before 1950)
        if timestep >= iterations - num_samples:
            # sampling
            samples = fns.sample_birds(bird_matrix, sample_freq[timestep])

            # updating information based on sample
            # number of sampled birds with a syllable type
            sample_bps = fns.count_type(samples, vector_size)
            # assume each sampled bird has never been sampled before
            # thus, add the sampled syllables to the count of individual birds that have had that syllable type
            sample_unique_bps += sample_bps
            # store timestep the syllable first appeared
            first_sampled[(first_sampled == 0) & (sample_bps > 0)] = \
                timestep
            # increment timestep the syllable last appeared
            last_sampled[sample_bps > 0] = timestep + 1

            # save the first sample
            if timestep == iterations - num_samples:
                sampled_bird_counts_t1950 = sample_bps.copy()

            # updating information based on full bird matrix
            current_bps = fns.count_type(bird_matrix, vector_size)
            actual_lifetimes[current_bps > 0] += 1

        # new_frame = ax.imshow(bird_matrix, cmap='gray')
        #
        # frames.append([new_frame])

    # video = animation.ArtistAnimation(fig, frames, interval=100, blit=False,
    #                                   repeat_delay=1000)
    # if want_to_save:
    #     video.save(video_name)
    plt.close()

    # calculate lifespan of sampled syllables
    sampled_lifetimes = last_sampled - first_sampled
    sampled_bird_counts_t2017 = sample_bps.copy()
    bird_counts_t2017 = current_bps.copy()

    # save full matrix information
    np.savetxt('actual_lifetimes_starting_1950.csv', actual_lifetimes,
               delimiter=",")
    np.savetxt('bird_counts_t0.csv', bird_counts_t0, delimiter=",")
    np.savetxt('bird_counts_t1950.csv', bird_counts_t1950, delimiter=",")
    np.savetxt('bird_counts_t2017.csv', bird_counts_t2017, delimiter=",")
    np.savetxt('unique_birds_with_syllables.csv', unique_bps,
               delimiter=",")

    np.savetxt('sampled_lifetimes.csv', sampled_lifetimes, delimiter=",")
    np.savetxt('sampled_bird_counts_t1950.csv', sampled_bird_counts_t1950,
               delimiter=",")
    np.savetxt('sampled_bird_counts_t2017.csv', sampled_bird_counts_t2017,
               delimiter=",")
    np.savetxt('sampled_birds_with_syllables.csv', sample_unique_bps,
               delimiter=",")


    print("--- %s seconds ---" % (time.time() - start_time))
