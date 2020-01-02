import numpy as np


def initiate(min_int, max_int, dim1, total_steps, vector_size):
    np.random.seed(49)
    # populate each element (territory) with a random bird (syllable type)
    # low (inclusive), high(exclusive), discrete uniform distribution
    bird_matrix = np.random.randint(min_int, max_int, [dim1, dim1],
                                    dtype='int')

    # initiate vectors for tracking info about entire population
    current_bps = np.zeros(vector_size, dtype='int')  # no. birds per syllable at one time step
    actual_lifetimes = np.zeros(vector_size, dtype='int')  # no. yrs syllable has existed
    unique_bps = np.zeros(vector_size, dtype='int')  # no. of individual birds that have ever had that syllable type


    # initiate vectors for tracking info about only the sampled population
    sample_bps = np.zeros(vector_size, dtype='int')
    sample_unique_bps = np.zeros(vector_size, dtype='int')
    first_sampled = np.zeros(vector_size, dtype='int')  # first iter syllable was sampled
    last_sampled = np.zeros(vector_size, dtype='int')  # last iter syllable was sampled + 1 (so can subtract to get lifespans)

    return bird_matrix, current_bps, actual_lifetimes, unique_bps, \
           sample_bps, sample_unique_bps, first_sampled, last_sampled


def count_type(territories, vector_size):
    unique, counts = np.unique(territories, return_counts=True)
    counts_array = np.zeros(vector_size, dtype='int')
    for u, c in zip(unique, counts):
        counts_array[u] = c
    return counts_array


def locate_dead_birds(num_loc, matrix_dim):
    loc_deaths = np.random.randint(0, matrix_dim, size=(num_loc, 2))  # by chance could get the same element twice
    return loc_deaths


def get_nearby_syllables(im, row, col, d=1):
    # does not wrap the boundaries (boundaries are real irl)

    # store value of dead bird's syllable
    dead_bird_syll = int(im[row, col])

    # to determine indices for surrounding neighbor squares
    row_start = row - d
    row_end = row + d + 1
    col_start = col - d
    col_end = col + d + 1

    # deal with edge/corner cases
    if row_start < 0:
        row_start = 0
    if row_end > len(im):
        row_end = len(im)
    if col_start < 0:
        col_start = 0
    if col_end > len(im):
        col_end = len(im)

    # get values of neighbors
    n = im[row_start:row_end, col_start:col_end].flatten().tolist()

    # remove syll of interest from list of neighboring sylls
    n.remove(dead_bird_syll)  # only removes first matching element

    return n


def get_learned_syll(sylls_to_copy, num_sylls, rule='neutral',
                     error_rate=None, direction=None):

    # if np.random.random() < error_rate:
    #     new_syll = num_sylls + 1
    #     num_sylls += 1
    if rule == 'neutral':  # a random nearby song
        new_syll = np.random.choice(sylls_to_copy)
    elif rule == 'conformity':  # most common value heard nearby, randomly chooses from ties
        new_syll = np.random.choice(np.where(np.bincount(sylls_to_copy) == np.bincount(sylls_to_copy).max())[0])
    elif rule == 'directional':  # value closest to some specified parameter, no need to handle ties (all same value)
        new_syll = sylls_to_copy[np.argmin(abs(sylls_to_copy - direction))]

    if np.random.random() < error_rate:
        new_syll = num_sylls + 1  # re-invention is not possible, always a new syllable type
        num_sylls += 1

    return new_syll, num_sylls


def sample_birds(all_territories, sampling_num):
    # get random matrix locations (x, y)
    loc_samples = np.random.randint(0, len(all_territories), size=(sampling_num, 2))  # by chance could get the same element twice
    sample_sylls = []
    for sample in loc_samples:
        # get syllable type of the sampled bird
        sample_sylls.append(all_territories[sample[0], sample[1]])

    return sample_sylls
