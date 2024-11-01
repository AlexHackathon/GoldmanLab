class EigenData:
    def __init__(self, value, vector):
        self.eigValue = value
        self.eigVect = vector
    def GetValue(self):
        return self.eigValue
    def GetVect(self):
        return self.eigVect
    def FindComp(self, listParam):
        if self.eigValue.imag == 0:
            return None
        for ed in listParam:
            val = ed.GetValue()
            if ed.real == self.eigValue.real and ed.imag == -self.eigValue.imag:
                return ed
        print("Can't find the pair")
        return None
    def IsRealOne(self, decimalPoints):
        return round(self.eigValue.real, decimalPoints) == 1.0
                
            
            
