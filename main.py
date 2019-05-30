import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# import seaborn as sns; sns.set()


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
    return loc_deaths


def get_nearby_syllables(im, row, col, d=1):
    # does not wrap the boundaries (boundaries are real irl)

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

    # set bird territory to non-syll value
    im[row, col] = -1

    # get values of neighbors
    n = im[row_start:row_end, col_start:col_end].flatten()
    # remove syll of interest from list of neighboring sylls
    values = n[n >= 0]

    return values


def get_learned_syll(sylls_to_copy, rule='neutral', error_rate=None, direction=None):

    if rule == 'neutral':  # a random nearby song
        new_syll = np.random.choice(sylls_to_copy)

    elif rule == 'conformity':  # most common value heard nearby, randomly chooses from ties
        new_syll = np.random.choice(np.where(np.bincount(sylls_to_copy) == np.bincount(sylls_to_copy).max())[0])

    elif rule == 'directional':  # value closest to some specified parameter, no need to handle ties (all same value)
        new_syll = sylls_to_copy[np.argmin(abs(sylls_to_copy - direction))]

    print('beginning syll', new_syll)

    # check if copy error is implemented
    if error_rate is None:
        insert = 'no'
    else:  # if some copy error, find if this bird has had an error while learning/invented a new syll
        r = np.random.random()
        print('error', r)
        if r < error_rate / 2:  # TODO make sure I have percentages correct with conditionals (equals to?)
            insert = 'left'
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
iterations = 100
low_prop = 0
high_prop = 100
dim = 20
mortality_rate = 0.4

bird_matrix, init_counts = initiate(low_prop, high_prop, dim, iterations)
total_territories = dim ^ 2
num_deaths = int(mortality_rate*total_territories)

# initialize figure for video frames
video_name = 'directionalTo49_5percent_100iters.mp4'
dpi = 100
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

frame = ax.imshow(bird_matrix, cmap='gray')
fig.set_size_inches([5, 5])
frames = [[frame]]

prop_counts = init_counts
for timestep in range(iterations):

    # some percent of birds die
    open_territories = locate_dead_birds(num_loc=num_deaths, matrix_dim=dim)

    new_props = []
    for bird in open_territories:
        # get new sylls for birds that will now occupy empty territories
        neighbor_sylls = get_nearby_syllables(bird_matrix, bird[0], bird[1], d=1)
        new_props.append(get_learned_syll(neighbor_sylls, rule='directional', error_rate=0.05, direction=high_prop // 2 - 1))

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

    frames.append([ax.imshow(bird_matrix, cmap='gray')])

video = animation.ArtistAnimation(fig, frames, interval=500, blit=False,
                                  repeat_delay=1000)
# video.save(video_name)
plt.close()

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
