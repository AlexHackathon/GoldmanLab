import random
students = [76,77,94,99,88,90,83,85,74,79,77,79,90,88,68,78,83,79,94,72,101,70,63,76,76,65,67,96,79,96]
row_count = 5
column_count = 6
rows = [[],[],[],[],[]]
used_students = []

def is_student_used(i):
    for s in used_students:
        if s == i:
            return True
    return False

for row in range(row_count):
    for i in range(column_count):
        student_used = True
        b = True
        while student_used:
            student = random.randint(0, len(students)-1)
            if not is_student_used(student):
                student_used = False
        rows[row].append(student)
        used_students.append(student)
print(rows)