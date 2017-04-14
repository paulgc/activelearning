"""Validation utilities"""

import pandas as pd


def validate_input_table(table, table_label):
    """Check if the input table is a dataframe."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(table_label + ' is not a dataframe')
    return True


def validate_attr(attr, table_cols, attr_label, table_label):
    """Check if the attribute exists in the table."""
    if attr not in table_cols:
        raise AssertionError(attr_label + ' \'' + attr + '\' not found in ' + \
                             table_label) 
    return True


def validate_attr_type(attr, attr_type, attr_label, table_label):
    """Check if the attribute is not of numeric type."""
    if attr_type != pd.np.object:
        raise AssertionError(attr_label + ' \'' + attr + '\' in ' + 
                             table_label + ' is not of string type.')
    return True


def validate_key_attr(key_attr, table, table_label):
    """Check if the attribute is a valid key attribute."""
    unique_flag = len(table[key_attr].unique()) == len(table)
    nan_flag = sum(table[key_attr].isnull()) == 0 
    if not (unique_flag and nan_flag):
        raise AssertionError('\'' + key_attr + '\' is not a key attribute ' + \
                             'in ' + table_label)
    return True


def validate_output_attrs(l_out_attrs, l_columns, r_out_attrs, r_columns):
    """Check if the output attributes exist in the original tables."""
    if l_out_attrs:
        for attr in l_out_attrs:
            if attr not in l_columns:
                raise AssertionError('output attribute \'' + attr + \
                                     '\' not found in left table')

    if r_out_attrs:
        for attr in r_out_attrs:
            if attr not in r_columns:
                raise AssertionError('output attribute \'' + attr + \
                                     '\' not found in right table')
    return True
