import numpy as np

count_type = {'lifespans', 'recordings'}


def collapse_array(array_to_collapse, data=None):
    if data not in count_type:
        raise ValueError('the type of array data must be specified '
                         'as either lifespans or recordings')

    array_collapsed = np.ones(len(array_to_collapse), dtype=int) * -1

    if data == 'lifespans':
        array_collapsed[0] = np.sum(array_to_collapse[0])
        array_collapsed[1] = np.sum(array_to_collapse[1:6])
        array_collapsed[6] = np.sum(array_to_collapse[6:11])
        array_collapsed[11] = np.sum(array_to_collapse[11:31])
        array_collapsed[31] = np.sum(array_to_collapse[31:51])
        array_collapsed[51] = np.sum(array_to_collapse[51:56])
        array_collapsed[56] = np.sum(array_to_collapse[56:61])
        array_collapsed[61] = np.sum(array_to_collapse[61:])
    elif data == 'recordings':
        array_collapsed[0] = np.sum(array_to_collapse[0])
        array_collapsed[1] = np.sum(array_to_collapse[1:6])
        array_collapsed[6] = np.sum(array_to_collapse[6:11])
        array_collapsed[11] = np.sum(array_to_collapse[11:16])
        array_collapsed[16] = np.sum(array_to_collapse[16:21])
        array_collapsed[21] = np.sum(array_to_collapse[21:26])
        array_collapsed[26] = np.sum(array_to_collapse[26:31])
        array_collapsed[31] = np.sum(array_to_collapse[31:])

    return array_collapsed
