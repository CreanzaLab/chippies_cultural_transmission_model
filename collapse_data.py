import numpy as np

count_type = {'lifespans', 'recordings'}


def collapse_array(array_to_collapse, data=None):
    if data not in count_type:
        raise ValueError('the type of array data must be specified '
                         'as either lifespans or recordings')

    array_collapsed = np.ones(len(array_to_collapse), dtype=int) * -1

    if data == 'lifespans':
        array_collapsed[0] = np.sum(array_to_collapse[0])
        array_collapsed[1] = np.sum(array_to_collapse[1:20])
        array_collapsed[20] = np.sum(array_to_collapse[20:35])
        array_collapsed[35] = np.sum(array_to_collapse[35:49])
        array_collapsed[49] = np.sum(array_to_collapse[49:57])
        array_collapsed[57] = np.sum(array_to_collapse[57:61])
        try:
            array_collapsed[61] = np.sum(array_to_collapse[61:])
        except IndexError:
            print('index 61 out of bounds')
    elif data == 'recordings':
        array_collapsed[0] = np.sum(array_to_collapse[0])
        array_collapsed[1] = np.sum(array_to_collapse[1:5])
        array_collapsed[5] = np.sum(array_to_collapse[5:10])
        array_collapsed[10] = np.sum(array_to_collapse[10:15])
        array_collapsed[15] = np.sum(array_to_collapse[15:22])
        array_collapsed[22] = np.sum(array_to_collapse[22:29])
        array_collapsed[29] = np.sum(array_to_collapse[29:])

    return array_collapsed
