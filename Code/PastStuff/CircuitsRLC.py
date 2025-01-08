t = input("What type: ")
r = int(input("R_eq at 0+: "))
l = int(input("L: "))
c = int(input("C: "))
if t == "s":
    a = r/(2*l)
    w = 1/((l*c)**(1/2))
    s1 = -a + (a**2 - w**2)**(1/2)
    s2 = -a - (a**2 - w**2)**(1/2)

    vc0 = int(input("vc0: "))
    vcI = int(input("vcI: "))
    ic0 = int(input("ic0: "))

    if a > w:

    elif a < w:

    else:
        
"""elif t == "p":
    a = 
    w = 
    
else:
    print("DNE")"""