import time
import numpy as np
from scipy.stats import linregress
import math
import random


def check_speed(function, r, N):
	"""
	These code gives some function with array input and measures time complexity of it with N max iterations and averaging by r - repeats
	"""
    logs = []

    for n in range(1, N + 1, 1):  
        t = 0
        for j in range(r):
            array = list(np.random.random((n)))

            start = time.time()
            function(array)
            end = time.time()
            #print(start, end)

            t += (end - start)
        logs.append(t / r)
    return logs

def check_speed_matrix(function, r, N):
	"""
	These code gives some function with array1 and array2 input and measures time complexity of it with N max
	iterations and averaging by r - repeats
	"""
    logs = []

    for n in range(1, N + 1, 1):
        t = 0
        for j in range(r):
            matrix1 = np.random.random((n, n))
            matrix2 = np.random.random((n, n))

            start = time.time()
            function(matrix1, matrix2)
            end = time.time()
            #print(start, end)

            t += (end - start)
        logs.append(t / r)
    return logs


# measaring time complexity for TASK#1
def const_function(array):
    array = 1
logs_1 = check_speed(const_function, 5, 2000)


# measaring time complexity for TASK#2 SumFunction
def sum_function(array):
  s = 0
  for i in range(len(array)):
      s += array[i]
logs_2 = check_speed(sum_function, 10, 2000)


# measaring time complexity for TASK#3 MultiplyFunction
def sum_function(array):
  s = 0
  for i in range(len(array)):
      s = s * array[i]
logs_3 = check_speed(sum_function, 10, 2000)


# measaring time complexity for TASK#4 P(1.0)
def get_p(array):
  s = 0
  for i in range(len(array)):
      s = s + (array[i] * math.pow(1.5, i))
logs_4 = check_speed(get_p, 10, 1700)


# measaring time complexity for TASK#5 Bubble sorting
def bubble(array):
    N = len(array)
    for i in range(N-1):
        for j in range(N-i-1):
            if array[j] > array[j+1]:
                  array[j], array[j+1] = array[j+1], array[j]
logs_5 = check_speed(bubble, 5, 1000)


# measaring time complexity for TASK#6 Quick sorting
def quicksort(nums):
   if len(nums) <= 1:
       return nums
   else:
       q = random.choice(nums)
       s_nums = []
       m_nums = []
       e_nums = []
       for n in nums:
           if n < q:
               s_nums.append(n)
           elif n > q:
               m_nums.append(n)
           else:
               e_nums.append(n)
       return quicksort(s_nums) + e_nums + quicksort(m_nums)
logs_6 = np.array(check_speed(quicksort, 5, 2000))


# measaring time complexity for TASK#7 Tim sorting
MIN_MERGE = 64

def calcMinRun(n):
    """Returns the minimum length of a
    run from 23 - 64 so that
    the len(array)/minrun is less than or
    equal to a power of 2.

    e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
    ..., 127=>64, 128=>32, ...
    """
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r


# This function sorts array from left index to
# to right index which is of size atmost RUN
def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1


# Merge function merges the sorted runs
def merge(arr, l, m, r):

    # original array is broken in two parts
    # left and right array
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])

    i, j, k = 0, 0, l

    # after comparing, we merge those two array
    # in larger sub array
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1

        else:
            arr[k] = right[j]
            j += 1

        k += 1

    # Copy remaining elements of left, if any
    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1

    # Copy remaining element of right, if any
    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1


# Iterative Timsort function to sort the
# array[0...n-1] (similar to merge sort)
def timSort(arr):
    n = len(arr)
    minRun = calcMinRun(n)

    # Sort individual subarrays of size RUN
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)

    # Start merging from size RUN (or 32). It will merge
    # to form size 64, then 128, 256 and so on ....
    size = minRun
    while size < n:

        # Pick starting point of left sub array. We
        # are going to merge arr[left..left+size-1]
        # and arr[left+size, left+2*size-1]
        # After every merge, we increase left by 2*size
        for left in range(0, n, 2 * size):

            # Find ending point of left sub array
            # mid+1 is starting point of right sub array
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))

            # Merge sub array arr[left.....mid] &
            # arr[mid+1....right]
            if mid < right:
                merge(arr, left, mid, right)

        size = 2 * size
logs_7 = np.array(check_speed(timSort, 5, 2000))


# measaring time complexity for TASK#8 Matrices multiplying
def matmult(a,b):
    zip_b = zip(*b)
    # uncomment next line if python 3 : 
    zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]
logs_8 = check_speed_matrix(matmult, 5, 1000)
