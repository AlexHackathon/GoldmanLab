"""
Problem 1
"""
def smallest(plist):
    #Base case: in a list of length one, the single element is the smallest
    if len(plist)==1:
        return plist[0]
    #Base case: in a list of length two, the smallest of the two is the smallest
    #Reductive step: the smallest in the list is always the smallest between the first element and the smallest of the rest of the list
    if len(plist) == 2:
        return min(plist[0],plist[1])
    else:
        return min(plist[0], smallest(plist[1:]))

"""
Problem 2 (linear version)
"""
def linearSearchValueIndexEqual(plist,idx=0):
    #An empty list never fits this description
    if len(plist) == 0:
        return []
    if len(plist) == 1:
        #A list of length 0 only agrees or doesn't agree with the criteria
        #Base case ends the recursion when the last index is searched
        if plist[0] == idx:
            return [plist[0]]
        else:
            return []
    else:
        #Check if the first index of the passed list agrees with the passed index
        if plist[0] == idx:
            #Add the remaining recursive calls to this value
            #Increase the index since index 0 now represents idx+1 (1 on second iteration)
            return [plist[0]]+linearSearchValueIndexEqual(plist[1:],idx+1)
        else:
            #Similar reductive call without adding because the first element didn't pass the test
            return linearSearchValueIndexEqual(plist[1:], idx+1)
"""
Problem 2 (binary version)
"""
def binarySearchValueIndexEqual(plist,myMin=1, myMax=0):
    #Sets the min and max the first time
    print(plist)
    if myMin == 1 and myMax ==0:
        myMin = 0
        myMax = len(plist)-1
    #Sets the middle to be the lower of 2 in an even case
    mid = (myMin+myMax)//2
    #An empty list never fits this description
    if len(plist) == 0:
        return []
    if len(plist)==1:
        if plist[len(plist)//2] == mid:
            return [plist[0]]
        else:
            return []
    else:
        if plist[(len(plist)-1)//2] > mid:
            #Keep left
            print("Too big")
            return binarySearchValueIndexEqual(plist[:(len(plist)-1)//2+1],0,mid)
        elif plist[(len(plist)-1)//2] < mid:
            #Keep the right
            print("Too small")
            return binarySearchValueIndexEqual(plist[(len(plist)-1)//2+1:],mid+1,myMax)
        else: #plist[mid] == mid
            return binarySearchValueIndexEqual(plist[:(len(plist)-1)//2+1],myMin,mid) + binarySearchValueIndexEqual(plist[(len(plist)-1)//2+1:],mid+1,myMax)
"""
Problem 3 (ladder)
"""

def ladder(rungs):
    #Any ladder with less than one rung can only be climbed one way
    if rungs <= 1:
        return 1
    else:
        #At each step you can go up one or two rungs
        return ladder(rungs-1) + ladder(rungs-2)
