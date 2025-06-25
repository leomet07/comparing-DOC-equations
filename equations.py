def get_1_4_ratio_and_2_4_ratio(bands):
    return (bands[0] / bands[3]), (bands[1] / bands[3])  # zero-indexed


def get_1_4_ratio_and_3_4_ratio(bands):
    return (bands[0] / bands[3]), (bands[2] / bands[3])  # zero-indexed


def get_3_4_ratio_and_4_5_ratio(bands):
    return (bands[2] / bands[3]), (bands[3] / bands[4])  # zero-indexed


equation_functions = [
    get_1_4_ratio_and_2_4_ratio,
    get_1_4_ratio_and_3_4_ratio,
    get_3_4_ratio_and_4_5_ratio,
]
