def slice_tuple(data_tuple, idx):
    tuple_ = []
    for data in data_tuple:
        tuple_.append(data[idx])
    return tuple(tuple_)