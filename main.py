import numpy as np
import matplotlib.pyplot as plt


def initiate(min_int, max_int, dim1, total_steps):
    # low (inclusive), high(exclusive), discrete uniform distribution
    bird_matrix = np.random.randint(min_int, max_int, [dim1, dim1])
    prop_types = np.zeros((max_int, total_steps))
    # init_counts = dict.fromkeys(list(range(low_prop, high_prop)), 0)
    return bird_matrix, prop_types


def count_type(prop_matrix, counts_array, step):
    unique, counts = np.unique(prop_matrix, return_counts=True)
    for u, c in zip(unique, counts):
        counts_array[u-1, step] = c
    # prop_counts = dict.fromkeys(list(range(10, 101)), 0)
    # prop_counts.update(dict(zip(unique, counts)))
    return counts_array


def locate_dead_birds(num_loc, matrix_dim):
    loc_deaths = np.random.randint(0, matrix_dim, size=(num_loc, 2))  # by chance could get the same element twice
    # print(loc_deaths)
    return loc_deaths


def get_new_prop(im, row, col, d=1, rule='neutral'):
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
        # print(values)
        new_syll = np.random.choice(values)
    elif rule == 'conformity':  # most common value heard nearby
        new_syll = np.argmax(np.bincount(values))
    elif rule == 'directional':
        new_syll = np.median(values)
    return new_syll



iterations = 100
low_prop = 0
high_prop = 100
dim = 20
mortality_rate = 0.4

bird_matrix, init_counts = initiate(low_prop, high_prop, dim, iterations)
total_territories = dim ^ 2
num_deaths = int(mortality_rate*total_territories)

for timestep in range(iterations):
    # some percent of birds die
    open_territories = locate_dead_birds(num_loc=num_deaths, matrix_dim=dim)
    new_props = []
    for bird in open_territories:
        # get new sylls for birds that will now occupy empty territories
        new_props.append(get_new_prop(bird_matrix, bird[0], bird[1], d=1, rule='conformity'))
    for bird, prop in zip(open_territories, new_props):
        # populate new territories
        bird_matrix[bird[0], bird[1]] = prop
    prop_counts = count_type(bird_matrix, init_counts, timestep)
    plt.imshow(bird_matrix)
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


# if row == 0:
# # neighbors1 = im[-1, col-d:col+d+1]
# # neighbors2 = im[row, col-d:col+d+1]
# # neighbors3 = im[row+d, col-d:col+d+1]
# # n = np.concatenate((neighbors1, neighbors2, neighbors3))
# wraps edge
# n = im[(-1, row, row + d), col - d:col + d + 1].flatten()
# values = np.hstack((n[:len(n)//2], n[len(n)//2+1:]))
# elif col == 0:
# wraps edge
# n = im[row - d:row + d + 1, (-1, col, col + d)].flatten()
# values = np.hstack((n[:len(n) // 2], n[len(n) // 2 + 1:]))