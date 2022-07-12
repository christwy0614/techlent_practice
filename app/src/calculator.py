class Calculator(): 
    def add(self, num1, num2):

        return num1 + num2
       
    def subtract(self, num1, num2):
        return num1 - num2

if __name__ == "__main__":
    calculator = Calculator()
    print(calculator.add(1,3))
    print(calculator.subtract(2,1))

