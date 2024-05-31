"""
Queue class used for Problem 1
"""
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

"""
Node class used for Problem 2-5
"""
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self, data):
        self.data = data

    def setNext(self, node):
        self.next = node



"""
1. Stack2 class
Implement stack data structure using queue
"""
class Stack2:
    def __init__(self):
        # Creates two queues to be used to create the stack
        self.q1 = Queue()
        self.q2 = Queue()
        return

    def isEmpty(self):
        # Returns the size of the stack by using the built in function of the Queue class
        return self.size()==0

    def push(self, item):
        #Push just acts as an enqueue for the Queue class.
        #pop() and peek() do most of the work for making the queues act as stacks
        self.q1.enqueue(item)

    def pop(self):
        #First, store all the elements other than the last one in another queue
        while self.q1.size() > 1:
            popped = self.q1.dequeue()
            self.q2.enqueue(popped)
        #Store the last entry of ans (Could be empty if the list is empty to begin with)
        ans = self.q1.dequeue()
        #Add everything from the second queue(everything other than the last item of the first queue) back into the first
        while self.q2.size() > 0:
            self.q1.enqueue(self.q2.dequeue())
        return ans

    def peek(self):
        #First, store all the elements other than the last one in another queue
        while self.q1.size() > 1:
            popped = self.q1.dequeue()
            self.q2.enqueue(popped)
        #Store the last entry of ans (Could be empty if the list is empty to begin with)
        ans = self.q1.dequeue()
        #Add it back into the second queue so that it isn't permanently removed
        self.q2.enqueue(ans)
        #Add everything from the second queue(everything other than the last item of the first queue) back into the first
        while self.q2.size() > 0:
            self.q1.enqueue(self.q2.dequeue())
        return ans

    def size(self):
        #Use the built in function of the Queue class on self.q1
        return self.q1.size()
"""
2. transform(lst)
Transform an unordered list into a Python list
Input: an (possibly empty) unordered list
Output: a Python list
"""
def transform(lst):
    curr = lst
    ans = []
    #return an empty list if there are no nodes
    if curr == None:
        return ans
    #Iterate through all the nodes adding the data to a python list
    while curr:
        ans.append(curr.getData())
        curr = curr.getNext()
    return ans



"""
3. concatenate(lst1, lst2)
Concatenate two unordered lists
Input: two (possibly empty) unordered list
Output: an unordered list
"""
def concatenate(lst1, lst2):
    #If any list is empty, return the head of the other list
    #If both lists are empty, return None
    #If both lists have nodes, continue to the iteration
    if lst1 == None:
        if lst2 == None:
            return None
        else:
            return lst2
    else:
        if lst2 == None:
            return lst1
    #Iterate through all the elements in the first list to get the last element
    curr = lst1
    while curr.getNext():
        curr = curr.getNext()
    #Set the next node of the last node to the first node of the second list
    curr.setNext(lst2)
    return lst1



"""
4. removeNodesFromBeginning(lst, n)
Remove the first n nodes from an unordered list
Input:
    lst -- an (possibly empty) unordered list
    n -- a non-negative integer
Output: an unordred list
"""
def removeNodesFromBeginning(lst, n):
    #Navigate to the element of index n (n+1 nth element)
    #Return it as the head
    curr = lst
    for i in range(n):
        curr = curr.getNext()
    return curr



"""
5. removeNodes(lst, i, n)
Starting from the ith node, remove the next n nodes
(not including the ith node itself).
Assume i + n <= lst.length(), i >= 0, n >= 0.
Input:
    lst -- an unordered list
    i -- a non-negative integer
    n -- a non-negative integer
Output: an unordred list

lst = [1, 2, 3, 4, 5]
i = 2
n = 2
return [1, 2, 5]

i = 1
n = 2
return [1, 4, 5]

i = 0
n = 2
return [3, 4, 5]
"""
def removeNodes(lst, i, n):
    curr = lst
    #If i==0 just call the removeNodesFromBeginning(n)
    if i == 0:
        return removeNodesFromBeginning(lst, n)
    #Goes to the last preserved node and saves it
    for j in range(i-1):
        curr = curr.getNext()
    connect = curr
    #Goes to the first node after the gap and sets the next node of the saved node to the current node
    for k in range(n+1):
        curr = curr.getNext()
    connect.next = curr
    return lst








