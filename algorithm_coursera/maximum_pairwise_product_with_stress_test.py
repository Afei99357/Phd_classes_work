# author: Eric Liao
from random import randint


def maximum_pairwise_product(integer_list):
    max_1 = 0
    max_1_index = 0
    max_2 = 0
    for i in range(len(integer_list)):
        if integer_list[i] > max_1:
            max_1 = integer_list[i]
            max_1_index = i

    for j in range(len(integer_list)):
        if integer_list[j] > max_2 and j != max_1_index:
            max_2 = integer_list[j]
    return max_1 * max_2


def maximum_pairwise_product_slow(integer_list):
    maximum_product = 0
    for i in range(len(integer_list)):
        for j in range(i + 1, len(integer_list)):
            if integer_list[i] * integer_list[j] > maximum_product:
                maximum_product = integer_list[i] * integer_list[j]
    return maximum_product


def main():
    ######## Here is the stress test #######
    # while True:
    #     number_list = []
    #     number_of_integer = randint(0, 100000000) % 1000 + 2
    #     for i in range(number_of_integer):
    #         n = randint(1, 10000000)
    #         number_list.append(n)
    #
    #     max_result_1 = maximum_pairwise_product(number_list)
    #     max_result_2 = maximum_pairwise_product_slow(number_list)
    #
    #     print(n)
    #     print(number_list)
    #     if max_result_1 == max_result_2:
    #         print("maximum result 1 is: " + str(max_result_1) + " and " + "maximum result 2 is: " + str(max_result_2))
    #         print("OK \n")
    #     else:
    #         print("maximum result 1 is: " + str(max_result_1) + " and " + "maximum result 2 is: " + str(max_result_2))
    #         print("Not OK \n")
    #         break

    try:
        number_list = []
        while True:
            number_list.append(
                int(input("Please enter the number, otherwise enter 'stop' and then click 'return: ")))

    except:
        print(number_list)

    print("The maximum pairwise product for the numbers you entered is: " + str(
        maximum_pairwise_product(number_list)))


if __name__ == '__main__':
    main()
