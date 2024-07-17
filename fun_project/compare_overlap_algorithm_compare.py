import random
import time
import matplotlib.pyplot as plt


def check_overplap(list1, list2):
    count = 0
    time_begin = time.time()
    list1.sort()
    list2.sort()
    index_list2_begin = 0
    for i in range(0, len(list1)):
        for j in range(index_list2_begin, len(list2)):
            if list1[i] == list2[j]:
                index_list2_begin = j + 1
                count = count + 1
                break
    time_end = time.time()
    return count, time_end - time_begin

def count_overlap_zipper(l1, l2):
    i = 0
    j = 0
    count = 0
    begin = time.time()
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            i += 1
        elif l1[i] > l2[j]:
            j += 1
        else:
            i += 1
            j += 1
            count += 1
    return count, time.time() - begin

def count_overlap_set(l1, l2):
    begin = time.time()
    count = len(set(l1) & set(l2))
    return count, time.time() - begin
number_of_elements = []
time_cost_list = []

def test_count_overlap():
    left = [1, 2, 2, 4, 5]
    right = [2, 2, 4, 7, 8]
    count, _ = count_overlap_zipper(left, right)
    assert count == 5

if __name__ == "__main__":
    ## 1000000, 10000000, 100000000, 1000000000, 10000000000
    for i in range(0, 100000000, 5000):
        list_1 = [random.randint(1, 1000000) for iter in range(i)]
        list_2 = [random.randint(1, 1000000) for iter in range(i)]
        number_of_elements.append(i)
        time_cost = count_overlap_set(list_1, list_2)
        time_cost_list.append(time_cost)
        print(i)
        print(time_cost)

    print("total time cost is: " + str(sum(time_cost_list)))

    plt.plot(number_of_elements, time_cost_list)
    plt.xlabel("Number of elments")
    plt.ylabel("Time cost")
    plt.show()
