from collections import *

"""
1. Tests whether the letters in a string can be permuted to form a palindrome.
"""
def canFormPalindrome(s):
    #Empty lists are palindromes
    if len(s) == 0:
        return True
    #Creates a dictionary with the number of each letter as a value
    numS = Counter(s)
    wasteMid = False
    #A combination is a palindrome if it has at most one odd
    #number of a letter
    for c in s:
        #Letter has already been checked
        if numS[c] == -1:
            continue
        #Letter has an odd number
        if numS[c]%2 == 1:
            #Allows for one odd number
            if wasteMid:
                return False
            else:
                wasteMid = True
        #Set it so that the letter has now been checked
        numS[c] = -1
    return True

"""
2. Determines if it is possible to write an anonymous letter using a book.
"""
def anonymousLetter(book, letter):
    #Keep a count of the number of letters in both strings
    numBook = Counter(book)
    numLetter = Counter(letter)
    for c in letter:
        #The book doesn't contain that character
        if numBook[c] == None:
            return False
        #Already checked if you had enough of this letter
        if numBook[c] == -1:
            continue
        #Checking if we have enough
        if numBook[c] < numLetter[c]:
            return False
        #Set the letter count so that you know you already checked the letter
        numLetter[c] = -1
    return True


