import numpy as np


def initiate(min_int, max_int, dim1, total_steps):
    np.random.seed(236)
    # populate each element (territory) with a random bird (syllable type)
    # low (inclusive), high(exclusive), discrete uniform distribution
    bird_matrix = np.random.randint(min_int, max_int, [dim1, dim1],
                                    dtype='int')

    # create array to store counts of syllable types for each iteration
    prop_types = np.zeros((max_int, total_steps + 1), dtype='int')
    prop_types = count_type(bird_matrix, prop_types, 0)
    # np.random.seed()
    return bird_matrix, prop_types


def count_type(prop_matrix, counts_array, step):
    unique, counts = np.unique(prop_matrix, return_counts=True)
    # print(unique)
    for u, c in zip(unique, counts):
        counts_array[u, step] = c
    return counts_array


def locate_dead_birds(ordered_pairs, num_loc):
    np.random.shuffle(ordered_pairs)
    loc_deaths = ordered_pairs[:num_loc]

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


def get_learned_syll(sylls_to_copy, rule='neutral', error_rate=None, direction=None):

    if rule == 'neutral':  # a random nearby song
        new_syll = np.random.choice(sylls_to_copy)

    elif rule == 'conformity':  # most common value heard nearby, randomly chooses from ties
        new_syll = np.random.choice(np.where(np.bincount(sylls_to_copy) == np.bincount(sylls_to_copy).max())[0])

    elif rule == 'directional':  # value closest to some specified parameter, no need to handle ties (all same value)
        new_syll = sylls_to_copy[np.argmin(abs(sylls_to_copy - direction))]

    # print('beginning syll', new_syll)

    # check if copy error is implemented
    if error_rate is None:
        insert = 'no'
    else:  # if some copy error, find if this bird has had an error while learning/invented a new syll
        r = np.random.random()
        # print('error', r)
        if r < error_rate / 2:
            insert = 'left'
        elif r > 1 - (error_rate / 2):
            insert = 'right'
            new_syll = new_syll + 1
        else:
            insert = 'no'
    # print('after error', new_syll, insert)

    return [new_syll, insert]


def sample_birds(all_territories, sampling_num):
    # get random matrix locations (x, y)
    loc_samples = np.random.randint(0, len(all_territories), size=(sampling_num, 2))  # by chance could get the same element twice
    sample_sylls = []
    for sample in loc_samples:
        # get syllable type of the sampled bird
        sample_sylls.append(all_territories[sample[0], sample[1]])

    return sample_sylls
