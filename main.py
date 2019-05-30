import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import cv2




def initiate(min_int, max_int, dim1, total_steps):
    # populate each element (territory) with a random bird (syllable type)
    # low (inclusive), high(exclusive), discrete uniform distribution
    bird_matrix = np.random.randint(min_int, max_int, [dim1, dim1])

    # create array to store counts of syllable types for each iteration
    prop_types = np.zeros((max_int, total_steps + 1))
    prop_types = count_type(bird_matrix, prop_types, 0)
    return bird_matrix, prop_types


def count_type(prop_matrix, counts_array, step):
    unique, counts = np.unique(prop_matrix, return_counts=True)
    print(unique)
    for u, c in zip(unique, counts):
        counts_array[u, step] = c
    return counts_array


def locate_dead_birds(num_loc, matrix_dim):
    loc_deaths = np.random.randint(0, matrix_dim, size=(num_loc, 2))  # by chance could get the same element twice
    # print(loc_deaths)
    return loc_deaths


def get_new_prop(im, row, col, d=1, rule='neutral', error_rate=None, direction=None):
    # does not wrap the boundaries

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

    # store syll value
    syll = im[row, col]
    # set bird territory to non-syll value
    im[row, col] = -1

    # take territory with neighbors
    n = im[row_start:row_end, col_start:col_end].flatten()
    # make list of neighbors syllables, not including bird of interest
    values = n[n >= 0]

    # reset bird of interest's syllable in the overall territory matrix
    im[row, col] = syll

    print(row, col, values)

    if rule == 'neutral':  # a random nearby song
        new_syll = np.random.choice(values)

    elif rule == 'conformity':  # most common value heard nearby, randomly chooses from ties
        new_syll = np.random.choice(np.where(np.bincount(values) == np.bincount(values).max())[0])

    elif rule == 'directional':  # value closest to some specified parameter, no need to handle ties (all same value)
        new_syll = values[np.argmin(abs(values-direction))]

    print('beginning syll', new_syll)
    # check if copy error is implemented
    if error_rate is None:
        insert = 'no'
    else:  # if some copy error, find if this bird has had an error while learning/invented a new syll
        r = np.random.random()
        print('error', r)
        if r < error_rate / 2:  # TODO make sure I have percentages correct with conditionals (equals to?)
            insert = 'left'
            # new_syll = new_syll
        elif r > 1 - (error_rate / 2):
            insert = 'right'
            new_syll = new_syll + 1
        else:
            insert = 'no'
    print('after error', new_syll, insert)

    return [new_syll, insert]


"""
Cultural Transmission Model
"""
iterations = 10
low_prop = 0
high_prop = 100
dim = 20
mortality_rate = 0.4

bird_matrix, init_counts = initiate(low_prop, high_prop, dim, iterations)
total_territories = dim ^ 2
num_deaths = int(mortality_rate*total_territories)

prop_counts = init_counts
for timestep in range(iterations):

    # some percent of birds die
    open_territories = locate_dead_birds(num_loc=num_deaths, matrix_dim=dim)

    new_props = []
    for bird in open_territories:
        # get new sylls for birds that will now occupy empty territories
        new_props.append(get_new_prop(bird_matrix, bird[0], bird[1],
                                      d=1, error_rate=0.5, rule='neutral', direction=None))

    # before adding new birds, shift syllable values (bird matrix and prop count) to make room for invented syllables
    unique_props = [list(x) for x in set(tuple(x) for x in new_props)]
    print(new_props)
    print('unique props', unique_props)
    for new in unique_props:
        print('item in unique props', new)
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

    # add new birds to the open territories (where old birds died)
    for bird, prop in zip(open_territories, [item[0] for item in new_props]):
        bird_matrix[bird[0], bird[1]] = prop

    # count syllables
    prop_counts = count_type(bird_matrix, prop_counts, timestep + 1)
    # plt.imshow(bird_matrix)
    # plt.show()

plt.bar(range(len(init_counts)), init_counts[:, 0])
plt.xlabel('syllable type')
plt.ylabel('number of birds with syllable type')
plt.title('random initial song types: discrete uniform distribution')
plt.show()

plt.hist(prop_counts[:, 0])
plt.title('beginning')
plt.xlabel('number of birds with a syllable type')
plt.ylabel('number of syllable types')
plt.show()

plt.hist(prop_counts[:, -1])
plt.title('end')
plt.xlabel('number of birds with a syllable type')
plt.ylabel('number of syllable types')
plt.show()
# print(bird_matrix, prop_counts)