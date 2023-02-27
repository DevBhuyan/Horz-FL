import gc
gc.enable()

def fast_non_dominated_sort(df_copy):
    l = df_copy.shape[0]
    df_copy = df_copy.sort_values('FCMI').reset_index(drop = True)
    fronts = []
    
    for sweep in range(l):
        rem = len(df_copy)
        if rem:
            front = []
            prev = df_copy.iloc[rem-1, :]
            front.append(prev)
            df_copy.drop(index = rem-1, inplace = True)
            df_copy.reset_index(drop = True, inplace = True)
            for i in range(rem-2, -1, -1):
                feature_tuple = df_copy.iloc[i, :]
                if feature_tuple['aFFMI'] < prev['aFFMI']:
                    front.append(feature_tuple)
                    prev = feature_tuple
                    df_copy.drop(index = i, inplace = True)
                    df_copy.reset_index(drop = True, inplace = True)
            fronts.append(front)   
        else:
            break
    return fronts

# Main program starts here
def nsga_2(dataset, df):
    
    # Initialization
    df_copy = df.copy(deep = True)
    
    # assert solution_fcmi == solution_fcmi
    # assert solution_affmi == solution_affmi
    
    non_dominated_sorted_solution = fast_non_dominated_sort(df_copy)

    ftrs_in_fronts = []
    for front in non_dominated_sorted_solution:
        ftrs_in_front = []
        for df in front:
            ftrs_in_front.append(df['features'])
        ftrs_in_fronts.append(ftrs_in_front)
        
    print('Pareto fronts: ', ftrs_in_fronts)
        
    return ftrs_in_fronts