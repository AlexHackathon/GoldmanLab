#Alexandru Tapus
#https://stackoverflow.com/questions/1740726/turn-string-into-operator (Used for a quick change from string to operator
#https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python (Used to convert strings to floats)
import operator
ops = {
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    '/' : operator.truediv,
}

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.items:
            return self.items.pop()
        else:
            print("pop() error: Stack is empty.")
            return None

    def peek(self):
        if self.items:
            return self.items[-1]
        else:
            print("peek() error: Stack is empty.")
            return None

    def size(self):
        return len(self.items)

def isFloat(s):
    #Tries to convert a string to a float
    #If it fails, it must not have been a float so it returns False
    try:
        float(s)
        return True
    except:
        return False
"""
1. postfixEval
Input type: a list of strings
Output type: a floating point number
"""
def postfixEval(expr):
    """Evaluate postfix notation"""
    numStack = Stack()
    num2 = None
    num1 = None
    #Iterate through the expression
    for i in range(len(expr)):
        if isFloat(expr[i]) or expr[i][0] == "-" and isFloat(expr[i][1:]): #Adds numbers that are strings on top of the stack (as ints)
            numStack.push(float(expr[i]))
            print(numStack.peek())
        elif expr[i] in ops.keys(): #Checks for whether the expression isn't a number but is an operator
            #Look at the last two numbers
            num2 = numStack.pop()
            print(num2)
            num1 = numStack.pop()
            print(num1)
            #Looks up which operation the string represents
            #Executes it on num1 and num2 and adds the result to the stack
            numStack.push(ops[expr[i]](num1, num2))
            print(numStack.peek())
        else: #Lumps together all error cases where the string isn't a number or an operator
            print("Invalid input")
            return None
    return numStack.peek()
"""
2. validParentheses
Input type: a string
Output type: a Boolean
"""
def validParentheses(s):
    """checks whether a string has proper closed parentheses"""
    #Relies on the idea that a parentheses must be opened before it can be closed
    stackLeft = Stack()
    for i in range(len(s)):
        #Adds all left parentheses to the stack as they are encountered
        if s[i] == "(" or s[i] == "[" or s[i] == "{":
            stackLeft.push(s[i])
            print(s[i])
        #When encountering a right parenthesis
        if s[i] == ")" or s[i] == "]" or s[i] == "}":
            print("Checking")
            print(s[i])
            #Creates a pair between the last seen open parenthesis and the current closed one
            parenPair = (stackLeft.peek(), s[i])
            if  parenPair == ("{","}") or parenPair == ("[","]") or parenPair == ("(",")"): #Removes a matched parenthesis from the stack and proceeds with further checks
                print("Good")
                stackLeft.pop()
            else: #It is an invalid string
                print("Not right")
                return False
    if stackLeft.size() != 0:
        return False
    return True



"""
3. reverseString
Input type: a string
Output type: a string
"""
def reverseString(s):
    """Write your code here"""
    myStack = Stack() #Defines a stack
    myString = "" #Creates an empty string to store the answer
    #Push the string into a stack
    for i in range(len(s)):
        myStack.push(s[i])
    #Pop the string out of the stack reversing it's order
    #Store the popped characters in the answer string
    for j in range(len(s)):
        myString = myString + myStack.pop()
    return myString
        
        







