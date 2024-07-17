import pandas as pd
import math


def gini_index(p1, p2, p3):
    gini_1 = 1 - pow(p1, 2) - pow(1 - p1, 2)
    gini_2 = 1 - pow(p2, 2) - pow(1 - p2, 2)
    gini_3 = 1 - pow(p3, 2) - pow(1 - p3, 2)

    gini_dic = {'node_1_gini': gini_1, 'node_2_gini': gini_2, 'node_3_gini': gini_3}

    return gini_dic


def entropy(p1, p2, p3):
    entropy_1 = - (p1 * math.log2(p1)) - ((1 - p1) * math.log2(1 - p1))
    entropy_2 = - (p2 * math.log2(p2)) - ((1 - p2) * math.log2(1 - p2))
    entropy_3 = - (p3 * math.log2(p3)) - ((1 - p3) * math.log2(1 - p3))
    entropy_dic = {'node_1_entropy': entropy_1, 'node_2_entropy': entropy_2, 'node_3_entropy': entropy_3}
    return entropy_dic


def misclassification_error(p1, p2, p3):
    error_1 = 1 - max(p1, 1 - p1)
    error_2 = 1 - max(p2, 1 - p2)
    error_3 = 1 - max(p3, 1 - p3)
    error_dic = {'node_1_error': error_1, 'node_2_error': error_2, 'node_3_error': error_3}
    return error_dic


# def gain(p1, p2, p3, gini_index_dictionary):
#     M1 = (10/10) * gini_index_dictionary['node_1_gini'] + (1-p1) * gini_index_dictionary['node_2_gini']
#     M2 = 4 / ((1-p1) * 10) * gini_index_dictionary['node_3_gini'] + 3 / ((1-p1) * 10) * gini_index_dictionary['node_2_gini']
#     M3 =

def main():
    df = pd.read_csv('/Users/ericliao/Desktop/phD_courses/2021Spring_class/data_Mining_TA/assignment_1/decisioin_tree_dataset.csv',
                     header=0, index_col=0)
    sample_number = 10

    df_node_1 = df[df['Home Owner'] == 'No']
    number_home_owner_yes = 10 - df_node_1['Home Owner'].size
    p_home_owner_yes = number_home_owner_yes / sample_number

    df_node_2 = df_node_1[df_node_1['Marital Status'] == 'Married']
    # df_node_2 = pd.concat([df_node_1.loc[(df_node_1['Marital Status'] == 'Single')], df_node_1.loc[(df_node_1['Marital Status'] == 'Divorced')]])
    number_marital_status_married = df_node_2['Marital Status'].size
    p_marital_status_married = number_marital_status_married / (sample_number - number_home_owner_yes)

    df_node_3 = pd.concat([df_node_1.loc[(df_node_1['Marital Status'] == 'Single')], df_node_1.loc[(df_node_1['Marital Status'] == 'Divorced')]])
    df_3 = df_node_3[df_node_3['Annual Income'] > 80000]['Annual Income']
    number_income_over_80K = df_3.size
    p_income_over_80K = number_income_over_80K / df_node_3['Marital Status'].size

    gini_index_dictionary = gini_index(p_home_owner_yes, p_marital_status_married, p_income_over_80K)
    entropy_dictionary = entropy(p_home_owner_yes, p_marital_status_married, p_income_over_80K)
    misclassification_error_dictionary = misclassification_error(p_home_owner_yes, p_marital_status_married,
                                                                 p_income_over_80K)

    # gain(p_home_owner_yes, p_marital_status_no_married, p_income_over_80K, gini_index_dictionary)

    columns = ['gini_index', 'entropy', 'misclassification error']

    data = [[gini_index_dictionary['node_1_gini'], entropy_dictionary['node_1_entropy'],
             misclassification_error_dictionary['node_1_error']],
            [gini_index_dictionary['node_2_gini'], entropy_dictionary['node_2_entropy'],
             misclassification_error_dictionary['node_2_error']],
            [gini_index_dictionary['node_3_gini'], entropy_dictionary['node_3_entropy'],
             misclassification_error_dictionary['node_3_error']]]

    index_name = ['Home Owner', 'Marital Status', 'Annual Income']
    output_df = pd.DataFrame(data, columns=columns, index=index_name)

    output_df.to_csv('/Users/ericliao/Desktop/phD_courses/2021Spring_class/data_Mining_TA/assignment1_2.csv')


if __name__ == '__main__':
    main()
