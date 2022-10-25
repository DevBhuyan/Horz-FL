# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

# Importing required modules
import math
import random
import pandas as pd
import matplotlib.pyplot as plt


# Function to find index of list

def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] < values2[q]) or (
                    values1[p] >= values1[q] and values2[p] < values2[q]) or (
                    values1[p] > values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] < values2[p]) or (
                    values1[q] >= values1[p] and values2[q] < values2[p]) or (
                    values1[q] > values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
                # print("n[p]", n[p])
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


# Function to calculate crowding distance


def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
    return distance


# Function to carry out the crossover


def crossover(a, b, min_x, max_x):
    r = random.random()
    if r > 0.5:
        return mutation((a + b) / 2, min_x, max_x)
    else:
        return mutation((a - b) / 2, min_x, max_x)


# Function to carry out the mutation operator
def mutation(solution, min_x, max_x):
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_x + (max_x - min_x) * random.random()
    return solution


# First function to optimize


def func_fcmi(x):
    return x


# Second function to optimize


def func_affmi(x):
    return x


# Main program starts here


def nsga_2(df):
    pop_size = 20
    max_gen = 921

    # Initialization
    min_x = -55
    max_x = 55
    solution_fcmi = df['FCMI'].to_list()
    solution_affmi = df['FFMI'].to_list()
    print(solution_fcmi)
    print(solution_affmi)
    # press_key = input("press any key to continue ")
    gen_no = 0
    while gen_no < max_gen:
        function1_values = [func_fcmi(solution_fcmi[i]) for i in range(0, pop_size)]
        function2_values = [func_affmi(solution_affmi[i]) for i in range(0, pop_size)]
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        # print("The best front for Generation number ", gen_no, " is")
        # for valuez in non_dominated_sorted_solution[0]:
            # print(round(solution_fcmi[valuez], 3), end=" ")
            # print(round(solution_affmi[valuez], 3), end=" ")
        # print("\n")
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
        solution2 = solution_fcmi[:]
        # Generating offsprings
        while len(solution2) != 2 * pop_size:
            a1 = random.randint(0, pop_size - 1)
            b1 = random.randint(0, pop_size - 1)
            solution2.append(crossover(solution_fcmi[a1], solution_affmi[b1], min_x, max_x))
        function1_values2 = [func_fcmi(solution2[i]) for i in range(0, 2 * pop_size)]
        function2_values2 = [func_affmi(solution2[i]) for i in range(0, 2 * pop_size)]
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == pop_size:
                    break
            if len(new_solution) == pop_size:
                break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    # Lets plot the final front now
    print("Global_FCMI :", function1_values)
    print("Global_aFFMI :", function2_values)
    df1 = pd.DataFrame({'FCMI': function1_values, 'FFMI': function2_values})
    print("df1",df1)
    df2 = pd.merge(df, df1['FCMI'], on=['FCMI'], how='inner')
    df2.drop(df2.index[df2['FCMI'] == 0.0], inplace=True)
    print("global feature list", df2)
    # function1 = [i * 1 for i in function1_values]
    # function2 = [j * 1 for j in function2_values]
    plt.xlabel('Maximize FCMI', fontsize=15)
    plt.ylabel('Minimize aFFMI', fontsize=15)
    plt.scatter(function1_values, function2_values)
    plt.show()

    return df2