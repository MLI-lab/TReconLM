
def print_dict(dict):

    """
    print a dictionary

    Args:
    dict (dict): The dictionary.
    """

    for key, value in dict.items():
        print(key, ":", value)


def print_list(list):

    """
    prints a list of strings (for each string in the list, a new row is printed)

    Args:
    list (list): The list of strings.
    """

    length = len(list)
    for i in range(length):
        print(list[i]+'\n')


if __name__ == "__main__":
     
     print("print_functions.py")
