import tkinter
import random
info_array = [["NH4","1","+","Ammonium"],
              ["H3O","1","+","Hydronium"],
              ["Ag","1","+","Silver"],
              ["Zn","2","+","Silver"],
              ["CH3COO","1","-","Acetate"],
              ["BrO","1","-","Hypobromite"],
              ["BrO2","1","-","Bromite"],
              ["BrO3","1","-","Bromate"],
              ["BrO4","1","-","Perbromate"],
              ["ClO","1","-","Hypochlorite"],
              ["ClO2","1","-","Chlorite"],
              ["ClO3","1","-","Chlorate"],
              ["ClO4","1","-","Perchlorate"],
              ["CN","1","-","Cyanide"],
              ["H2PO4","1","-","Dihydrogen Phosphate"],
              ["HCOO","1","-","Formate"],
              ["HCO3","1","-","Hydrogen Carbonate"],
              ["HSO4","1","-","Hydrogen Sulfate"],
              ["HSO3","1","-","Hydrogen Sulfite"],
              ["HS","1","-","Bisulfate"],
              ["OH","1","-","Hydroxide"],
              ["NO2","1","-","Nitrite"],
              ["NO3","1","-","Nitrate"],
              ["IO","1","-","Hypoiodite"],
              ["IO2","1","-","Iodite"],
              ["IO3","1","-","Iodate"],
              ["IO4","1","-","Periodate"],
              ["MnO4","1","-","Permanganate"],
              ["SCN","1","-","Thiocyanate"],
              ["CO3","2","-","Carbonate"],
              ["CrO4","2","-","Chromate"],
              ["Cr2O7","2","-","Dichromate"],
              ["HPO4","2","-","Hydrogen Phosphate"],
              ["C2O4","2","-","Oxalate"],
              ["O2","2","-","Peroxide"],
              ["SO3","2","-","Sulfite"],
              ["SO4","2","-","Sulfate"],
              ["S2O3","2","-","Thiosulfate"],
              ["PO3","3","-","Phosphite"],
              ["PO4","3","-","Phosphate"],
              ["AsO4","3","-","Arsenate"],
              ]

class Compound:
    def __init__(self, info):
        self.name = info[3]
        self.charge = info[1]
        self.sign = info[2]
        self.formula = info[0]
        self.fullFormula1 = self.formula + self.charge + self.sign
        self.fullFormula2 = self.formula + self.sign + self.charge
        if self.charge == "1":
            self.fullFormulaNoCharge = self.formula + self.sign
        self.chargeAns1 = self.charge + self.sign
        self.chargeAns2 = self.sign + self.charge
        if self.charge == "1":
            self.chargeAns3 = self.sign
while True:
    idx = random.randint(0,len(info_array)-1)
    question_type = random.randint(0,2)
    if question_type == 0:
        #Give the chemical formula and ask the charge
        correctAns = Compound(info_array[idx])
        ans = input("What is the charge of the " + correctAns.formula + " ion: ")
        match ans:
            case correctAns.chargeAns1:
                print("Correct!")
            case correctAns.chargeAns2:
                print("Correct!")
            case "break":
                break
            case _:
                if hasattr(correctAns, 'chargeAns3'):
                    if(ans == correctAns.chargeAns3):
                        print("Correct")
                    else:
                        print("The charge of the " + correctAns.formula + " ion is " + correctAns.chargeAns1)
                else:
                    print("The charge of the " + correctAns.formula + " ion is " + correctAns.chargeAns1)
    elif question_type == 1:
        #Give the name and ask for the whole formula (0+1 or 0+space+1)
        correctAns = Compound(info_array[idx])
        ans = input("What is the formula of the " + correctAns.name + " ion: ")
        match ans:
            case correctAns.fullFormula1:
                print("Correct!")
            case correctAns.fullFormula2:
                print("Correct!")
            case "break":
                break
            case _:
                if hasattr(correctAns, 'fullFormulaNoCharge'):
                    if(ans == correctAns.fullFormulaNoCharge):
                        print("Correct")
                    else:
                        print("The formula of the " + correctAns.name + " ion is " + correctAns.fullFormula1)
                else:
                    print("The formula of the " + correctAns.name + " ion is " + correctAns.fullFormula1)
    elif question_type == 2:
        #Give the chemical formula and ask the name
        correctAns = Compound(info_array[idx])
        ans = input("What is the name of the " + correctAns.fullFormula1 + " ion: ")
        if(ans == correctAns.name):
            print("Correct")
        elif(ans == "break"):
            break
        else:
            print("The name of the " + correctAns.fullFormula1 + " ion is " + correctAns.name)
print("Session ended.")

