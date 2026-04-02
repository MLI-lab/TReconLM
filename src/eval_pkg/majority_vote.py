
from collections import Counter
from typing import List
import numpy as np
import random
import operator

import sys


def majority_vote(list, length, check_length):

    """
    computes a positon wise majority vote over a list of strings

    Args:
    list (list): The list of strings, possibly of different length.
    length (int): The length of the result string 
    check_length (int): The intended length of the lists.

    Returns:
    string: the result string.
    """

    # TODO: Do we need this here? 
    check_variable = False
    if check_variable and len(list) != check_length:
        print('Error: something went wrong - majority_vote')
        return -1

    # Trim the strings to the fixed length
    strs = [s[:length] for s in list]

    # Pad shorter strings with a special character
    strs = [s.ljust(length, '#') for s in strs]

    # Transpose the list of strings
    transposed = zip(*strs)

    # Find the most common character at each position, handling ties between '#' and another character
    most_common_chars = []
    for chars in transposed:
        counter = Counter(chars)
        most_common = counter.most_common(2)
        if most_common[0][0] in ['#', '-'] and len(most_common) > 1:
            # The most common character is '#' or '-' and there is another character with the same count, so choose the other character
            most_common_chars.append(most_common[1][0])
        else:
            # The most common character is not '#' or '-' or there is no other character with the same count, so add the most common character
            most_common_chars.append(most_common[0][0])

    # Join these characters together to form the final string
    result = ''.join(most_common_chars)
    return result


def simple_majority_vote(list, length, check_length):
    """
    Computes a position wise majority vote over a list of strings

    Args:
    list (list): The list of strings, possibly of different length.
    length (int): The length of the result string 
    check_length (int): The intended length of the lists.

    Returns:
    string: the result string.
    """

    # TODO: Do we need this here? 
    check_variable = False
    if check_variable and len(list) != check_length:
        print('Error: something went wrong - majority_vote')
        return -1

    # Trim the strings to the fixed length
    strs = [s[:length] for s in list]

    # Transpose the list of strings
    transposed = zip(*strs)

    # Find the most common character at each position, making a random decision in case of a tie
    most_common_chars = []
    for chars in transposed:
        counter = Counter(chars)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        # Get all characters with the maximum count
        max_chars = [char for char, count in most_common if count == max_count]
        # Choose a random character from max_chars
        most_common_chars.append(random.choice(max_chars))

    # Join these characters together to form the final string
    result = ''.join(most_common_chars)

    return result


from collections import Counter

def majority_vote_new(list, length, check_length):
    """
    Computes a position-wise majority vote over a list of strings.

    Args:
    list (list): The list of strings, possibly of different lengths.
    length (int): The length of the result string.
    check_length (int): The intended length of the lists.

    Returns:
    string: The result string.
    """

    print_flag = False

    # Trim the strings to the fixed length
    strs = [s[:length] for s in list]

    # Pad shorter strings with a special character
    strs = [s.ljust(length, '#') for s in strs]

    if print_flag:
        print('strs:', strs)

    # Transpose the list of strings
    transposed = zip(*strs)

    # Find the most common character at each position, handling ties between '#' and another character
    most_common_chars = []
    for chars in transposed:

        case = ''
    
        counter = Counter(chars)
        most_common = counter.most_common()
        
        if print_flag:
            print('----------------------------------------------------------------------------------')
            print(chars)
            print(counter)
            print(most_common)

        # Handling ties and prioritizing regular characters over '-'
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:

            # If there's a tie, check if one of the top two is not '-'
            if most_common[0][0] != '-' and most_common[1][0] != '-':
                case = 'case1.1'
                most_common_chars.append(most_common[0][0])  # Choose the first if both are regular characters
            elif most_common[0][0] == '-' and most_common[1][0] != '-':
                #print('case1.2')
                case = 'case1.2'
                most_common_chars.append(most_common[1][0])  # Choose the second if it's a regular character
            else:
                #print('case1.3')
                case = 'case1.3'
                most_common_chars.append(most_common[0][0])  # Choose the first (could be '-')
        else:
            # print('case2')  
            # No tie or a tie that doesn't involve '-'
            case = 'case2'
            most_common_chars.append(most_common[0][0])

        if print_flag:
            print(case)

        if case == '':
            print('Error: case is empty')
            sys.exit()

    # Join these characters together to form the final string
    result = ''.join(most_common_chars)
    return result

from collections import Counter

def majority_vote_v1(list, length, check_length):
    """
    Computes a position-wise majority vote over a list of strings.

    Args:
    list (list): The list of strings, possibly of different lengths.
    length (int): The length of the result string.
    check_length (int): The intended length of the lists.

    Returns:
    string: The result string.
    """

    print_flag = False

    # Trim the strings to the fixed length
    strs = [s[:length] for s in list]

    # Pad shorter strings with a special character
    strs = [s.ljust(length, '#') for s in strs]

    if print_flag:
        print('strs:', strs)

    # Transpose the list of strings
    transposed = zip(*strs)

    # Find the most common character at each position, handling ties between '#' and another character
    most_common_chars = []
    for chars in transposed:

        case = ''
    
        counter = Counter(chars)
        most_common = counter.most_common()
        
        if print_flag:
            print('----------------------------------------------------------------------------------')
            print(chars)
            print(counter)
            print(most_common)

        # Handling ties and prioritizing regular characters over '-'
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:

            # If there's a tie, check if one of the top two is not '-'
            if most_common[0][0] != '-' and most_common[1][0] != '-':
                case = 'case1.1'
                most_common_chars.append(most_common[0][0])  # Choose the first if both are regular characters
            elif most_common[0][0] == '-' and most_common[1][0] != '-':
                case = 'case1.2'
                most_common_chars.append(most_common[1][0])  # Choose the second if it's a regular character
            else:
                case = 'case1.3'
                most_common_chars.append(most_common[0][0])  # Choose the first (could be '-')
        else:
            case = 'case2'
            most_common_chars.append(most_common[0][0])

        # Adjust for 'D' character
        if most_common_chars[-1] == 'D':
            found = False
            for char, count in most_common:
                if char in 'ACTG':
                    most_common_chars[-1] = char
                    found = True
                    break
            if not found:
                print('random choice')
                most_common_chars[-1] = random.choice('ACTG')

        if print_flag:
            print(case)

        if case == '':
            print('Error: case is empty')
            sys.exit()

    # Join these characters together to form the final string
    result = ''.join(most_common_chars)
    return result

import sys


def majority_merge(reads, weight=0.4):
    # Assume all reads have the same length
    #print(f"Read of length {len(reads[0])}")
    #print("Voting per position:")
    #for i in range(len(reads[0])):
    #    # Print column i of all reads
    #    column = [read[i] for read in reads]
    #    print(f"Pos {i:3d}: {'  '.join(column)}")
    #    sys.stdout.flush()  
    res = ""
    for i in range(len(reads[0])):
        counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, '-': 0, 'N': 0}
        for read in reads:
            counts[read[i]] += 1
        counts['-'] *= weight
        mv = max(counts.items(), key=operator.itemgetter(1))[0]
        if mv != '-':
            res += mv
    return res



def extended_majority_vote(list, length, check_length):
    """
    Computes a position-wise majority vote over a list of strings.

    Args:
    list (list): The list of strings, possibly of different lengths.
    length (int): The length of the result string.
    check_length (int): The intended length of the lists.

    Returns:
    string: The result string.
    """
    max_len = max([len(s) for s in list])

    print_flag = False

    # Trim the strings to the fixed length
    #strs = [s[:length] for s in list]
    

    # Pad shorter strings with a special character
    strs = [s.ljust(max_len, '#') for s in list]

    if print_flag:
        print('strs:', strs)

    # Transpose the list of strings
    transposed = zip(*strs)

    # Find the most common character at each position, handling ties between '#' and another character
    most_common_chars = []
    for chars in transposed:

        counter = Counter(chars)
        most_common = counter.most_common()
        
        if 'I' in chars:
            most_common_chars.append('-')
        elif 'D' in chars:
            for char, count in most_common:
                if char != 'D' and char != '-':
                    most_common_chars.append(char)
                    break
        else:
            most_common_chars.append(most_common[0][0])
    # Join these characters together to form the final string
    result = ''.join(most_common_chars)
    return result

def weighted_majority_vote(list: List[str], length: int, check_length: int, weights: np.ndarray)->str:

    """
    Computes a position wise majority vote over a list of strings with weights

    Args:
    list (list): The list of strings, possibly of different length.
    length (int): The length of the result string
    check_length (int): The intended length of the lists.
    weights (list): The list of weights for each string.

    Returns:
    string: the result string.
    """

    if len(list) != check_length:
        raise ValueError("Length of list does not match check_length")

    result = ''

    

    for i in range(length): # iterate over column index i
        #print('i:', i)
        char_counts = {}
        for s, w in zip(list, weights): # iterate over rows
            if i < len(s):
                char_counts[s[i]] = char_counts.get(s[i], 0) + w[i]
                #print(char_counts)
        most_common_char = max(char_counts, key=char_counts.get)
        result += most_common_char

    return result


if __name__ == '__main__':

    print("Running majority_vote.py")
    # Test majority_vote
    list = ['ACGT-', 
            'A-GT-', 
            'T-G-G', 
            'T-G-G']
    
    check_length = len(list)
    length = len(list[0])

    
    # candidate = majority_vote_new(list, length = length, check_length = check_length)
    candidate = majority_merge(list, weight = 0.4)

    print(candidate)
    sys.exit()

    
    # Create a random numpy array of dimension 4x5
    array = np.random.rand(4, 5)

    # Create a numpy array of dimension 4x5 with specific values
    array = np.array([[0.3,  0.2, 0.3, 0.4, 0.5],
                      [0.9,  0.3, 0.4, 0.5, 0.1],
                      [0.25, 0.4, 0.5, 0.1, 0.2],
                      [0.25, 0.1, 0.2, 0.3, 0.4]])

    # Normalize the array so that the column sum adds up to one
    array /= array.sum(axis=0)
    print(array)

    length = 4
    check_length = 4
    result = majority_vote(list, length, check_length = 2)
    print('majority vote:', result) # Expected: ACGT

    result = weighted_majority_vote(list, length, check_length, array)
    print('weighted majority vote:', result) 