import random
info_array = [["Sn","2+4+","Tin"],
              ["Pb","2+3+4+","Lead"],
              ["V", "2+3+4+","Vanadium"],
              ["Cd","2+","Cadmium"],
              ["Cr","2+3+","Chromium"],
              ["Cu","1+2+","Copper"],
              ["Hg2/Hg", "2+", "Mercury"],
              ["Co","2+3+","Cobalt"],
              ["Fe","2+3+","Iron"],
              ["Mn","2+7+","Manganese"],
              ["Ni","2+3+","Nickel"]]
idx=0
correctCount=0
while idx < len(info_array):#True:
    #idx = random.randint(0,len(info_array)-1)
    question_type = random.randint(0,1)
    if question_type == 0:
        #Give the name and ask the charges
        ans = input("What are the charges of the " + info_array[idx][2] + " ion: ")
        if ans == info_array[idx][1]:
            print("Correct!")
            correctCount = correctCount + 1
        elif ans == "break":
            break
        else:
            print("The charges of the " + info_array[idx][2] + " ion are " + info_array[idx][1])
    elif question_type == 1:
        #Give the formula and ask the charges
        ans = input("What are the charges of the " + info_array[idx][0] + " ion: ")
        if ans == info_array[idx][1]:
            print("Correct!")
            correctCount = correctCount + 1
        elif ans == "break":
            break
        else:
            print("The charges of the " + info_array[idx][0] + " ion are " + info_array[idx][1])
    idx = idx + 1
print(correctCount/len(info_array) * 100 // 1)
#print("Session ended.")