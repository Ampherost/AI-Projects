# warmup.py
# ---------
# 
# Author: Ioannis Karamouzas (ioannis@cs.ucr.edu)
#


"""
  This code will allow you to verify the the successfull installation of 
  Python on your machine. If you want to practice a bit your Python skills, 
  please modify the functions below as indicated. 
"""

def max_index(data):
    return data.index(max(data))


def two_sum(nums, target):
    seen = set()
    for num in nums:
        complement = target - num
        if complement in seen:
            return complement, num
        seen.add(num)
    return None, None


def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)



if __name__ == "__main__":
    import sys
    print("Python {}".format(sys.version))
    assert(sys.version_info[0]==3 and (sys.version_info[1]==11 or sys.version_info[1]==12))
    import tkinter
    print("Tkinter {}".format(tkinter.TkVersion))
    import numpy as np
    print("Numpy {}".format(np.__version__))
    import matplotlib
    print("Matplotlib {}".format(matplotlib.__version__))
    
    c = input("\n"
        "Congrats! You have successfully installed Python.\n"
        "Please enter y to go on autograding or n to exit.\n"
        "[y/n] "
    )
    if c.lower() != "y": exit()
    
    data = list(np.random.rand(26))
    assert(data[max_index(data)] == max(data))

    nums = np.array([2, 3, 5, 7, 11, 13, 17, 19])
    np.random.shuffle(nums)
    tar = sum(nums[np.random.randint(0, len(nums), 2)])
    v1, v2 = two_sum(list(nums), tar)
    assert(v1 in nums and v2 in nums and v1+v2==tar)

    n = np.random.randint(3, 10)
    lookup = [
        1, 1, 2, 6, 24,
        120, 720, 5040, 40320, 362880
    ]
    assert(lookup[n] == factorial(n))
    print("Good job!")