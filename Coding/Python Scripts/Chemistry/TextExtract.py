startInd = 0
currInd = 0
text = "11.13 10.59 10.20 9.94 9.59 9.56 9.53 9.48 9.44 9.40 9.36 9.28 9.23 9.18 9.13 9.07 8.97 8.79 8.60 8.29 8.06 7.86 7.68 7.54 7.40 7.35 7.33 7.26 7.20 7.16 7.09 7.06 7.03 7.00 6.59 6.26 5.88 5.28 5.20 5.11 4.98 4.76 4.45 3.79 3.34 3.18 3.05 2.93 2.87 2.80 2.74 2.69 2.59  2.55 2.51 2.48 2.46 2.43 2.41 2.39 2.37 2.35 2.33 2.31 2.30 2.28 2.26 2.12 2.01 1.39"
while True:
    if len(text) == currInd:
        print(text[int(startInd):int(currInd)])
        break
    if text[currInd] == "x" or text[currInd] == "X" or text[currInd] == " ":
        print(text[int(startInd):int(currInd)])
        startInd = currInd + 1
        currInd = currInd + 1
    currInd = currInd + 1
        