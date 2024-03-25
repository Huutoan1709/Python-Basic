# import numpy as np
# import row as row
# import matplotlib.pyplot as plt
#
#
# def print_hi(fullname):
#     for c in fullname:
#         print(c)
#
#
# def odd_Number():
#     for i in range(1, 11):
#         if i % 2 != 0:
#             print(i, end=" ")
#
#
# # 3A tinh tong cac so trong bai 2
# def Tong_2():
#     sum = 0
#     for i in range(1, 11):
#         if i % 2 != 0:
#             sum += i
#     return sum
#
#
# # 3B tinh tong cac so tu 1 toi 6
# def Tong_3():
#     sum = 0
#     for i in range(1, 6):
#         sum += i
#     print("Tong cac so tu 1 toi 6 : ", sum)
#
#
# # 4
# def myDict():
#     print("\n\nCau 4:")
#     mydict = {"a": 1, "b": 2, "c": 3, "d": 4}
#     # Print all key in mydict
#     for i in mydict.keys():
#         print(i, end=" ")
#     # Print all values in mydict
#     for i in mydict.values():
#         print(i)
#     # Print all keys and values
#     for i in mydict.items():
#         print(i)
#
#
# # In cac khoa hoc tuong ung
# def My_Coureces():
#     courses = [131, 141, 142, 212]
#     names = ["Maths", "Physics", "Chem", "Bio"]
#     mycources = zip(courses, names)
#     print(list(mycources))
#
#
# # Cau 6
# # b
# def look_ueoai():
#     count = 0
#     keyy = "jabbawocky"
#     for i in keyy:
#         if i.lower() not in "ueoai":
#             count += 1
#     print("So phu am la :", count)
#
#     for i in keyy:
#         if i.lower() in "ueoai":
#             continue
#         print(i, end="")
#
#
# # cau7
# def print_a():
#     for a in range(-2, 3):
#         try:
#             result = 10 / a
#             print("\n10/", a, " = ", result)
#
#         except ZeroDivisionError:
#             print("khong the chia cho so 0")
#
#
# # cau8
# def Given_Age():
#     given_Ages = [23, 10, 80]
#     names = ["Hoa", "lam", "Nam"]
#     data = zip(given_Ages, names)
#     x = sorted(data, key=lambda x: x[0])
#
#     print(x)
#
#
# # cau9
#
# def input_file():
#     input_file = open("firstname.txt")
#     for line in input_file:
#         print(line, end='')
#     input_file.close()
#     # readline.es Doc them ki tu xuong dong
#     input_file = open("firstname.txt")
#     first_names = input_file.readlines()
#     input_file.close()
#     print("\n", first_names)
#
#
# def Matrixxx():
#     matrix = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
#     vector = np.array([1, 2, 3])
#     matrix_rank = np.linalg.matrix_rank(matrix)
#
#     print("Ma tran :",matrix)
#     print("Hang : ",matrix_rank)
#     print("Shape: ",matrix.shape)
#     new_matrix = 3 + matrix
#     print ("Ma Tran moi : ", new_matrix)
#
#     #ChuyenVi
#     vector_chuyenvi = np.transpose(vector)
#     matrix_chuyenvi = np.transpose(matrix)
#     print ("ma tran chuyen vi : ", matrix_chuyenvi)
#     print ("Vec to chuyen vi ; ", vector_chuyenvi)
#
# def Tonga_b(a, b):
#     return a + b
#
#
# #cau 5 Compute the norm of x=(2,7). Normalization vector x.
# def normvecto():
#     x = np.array([2,7])
#     norm_x = np.linalg.norm(x)
#     print ("Dinh muc la : ", norm_x)
#     normalized_x = x / norm_x
#     print ("normalized vector", normalized_x)
#
# #6/ Given a=[10,15], b=[8,2] and c=[1,2,3]. Compute a+b, a-b, a-c. Do all of them work? Why?
# def completecb():
#     a = np.array([10,15])
#     b = np.array([8, 2])
#     c = np.array([1,2,3])
#
#     print("Tong a+b = ", a+b)
#     print("Tong a-b = ", a - b)
#     try:
#         print("hieu a c ",a-c)
#     except ValueError as e:
#         print(e)
#
# #7/ Compute the dot product of a and b.
# def dot_product():
#     a = np.array([10, 15])
#     b = np.array([8, 2])
#     tichvohuong = np.dot(a ,b)
#     print("Tich vo huong 2 vector a b la : ",tichvohuong)
#
# #8/ Given matrix A=[[2,4,9],[3,6,7]].
# 	#a/ Check the rank and shape of A
# 	#b/ How can get the value 7 in A?
# 	#c/ Return the second column of A.
# def check_matrix():
#     A = np.array([[2, 4, 9], [3, 6, 7]])
#     print("Rank of A : ", np.linalg.matrix_rank(A))
#     print("Shape of A:  ", A.shape)
#     value_7_index = np.where(A == 7)
#     print("value 7 trong ma tran la : " ,A[value_7_index])
#     column_2 = A[:, 1]
#     print("Cot thu 2 cua ma tran A : ",column_2)
# #Create a random  3x3 matrix  with the value in range (-10,10).
# def rand_matrix():
#     matrix = np.random.randint(-10,10, size=(3,3))
#     print(matrix)
# def identity_matrix():
#     identity_Matrix = np.eye(3)
#     print(identity_Matrix)
#
# def func11():
#     matrix = np.random.randint(1,10, size=(3,3))
#     traceA = np.trace(matrix)
#     print("Vet cua ma tran: ", traceA)
#     traceB =0
#     for i in range(3):
#         traceB += matrix[i,i]
#     print("Vet cua ma tran khi dung loop: ", traceB)
# #cau 12 Create a 3x3 diagonal matrix with the value in main diagonal 1,2,3.
# def func12():
#     matrix = np.diag([1,2,3])
#     print("Ma trận với đường chéo chính : ", matrix)
#
# def func13():
#     A = np.array([[1,1,2],[2,4,-3],[3,6,-5]])
#     determinant_A = np.linalg.det(A)
#     print("Dinh thuc", determinant_A)
#
# def func14():
#     A1 = np.array([1,-2,-5])
#     A2 = np.array([2,5,6])
#     matrix = np.column_stack((A1,A2))
#     print(matrix)
#
# def func15():
#     y_values = range(-5, 6)
#
#     # Calculate the square of each y value
#     y_squared = [y ** 2 for y in y_values]
#
#     # Plot the values
#     plt.plot(y_values, y_squared)
#     plt.xlabel('y')
#     plt.ylabel('y squared')
#     plt.title('Plot of y squared')
#     plt.grid(True)
#     plt.show()
#
# def func16():
#     value = np.linspace(0,32,num=4)
#     print(value)
#
# def  func17():
#     x = np.linspace(-5, 5,num=50)
#     y = x**2
#     # Calculate the square of each y value
#
#     # Plot the values
#     plt.plot(x,y)
#     plt.xlabel('y')
#     plt.ylabel('y squared')
#     plt.title('Plot of y squared')
#     plt.grid(True)
#     plt.show()
# def func18():
#     x = np.linspace(-5, 5, num=100)
#     y = np.exp((x))
#     # Calculate the square of each y value
#
#     # Plot the values
#     plt.plot(x, y, label="y=exp(x)")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title('Plot of y squared')
#     plt.grid(True)
#     plt.show()
# #Plot y=exp(x) with label and title.
#
import csv


def show5_line():
    with open('04_gap-merged.tsv','r') as file:
        for i in range(5):
            line = file.readline().strip()
            print(line)


def find_number():
    num_rows =0
    num_cols=0
    with open('04_gap-merged.tsv', 'r') as file:
        for i in file:
            num_rows += 1
            num_cols = max(num_cols, len(i.strip().split('\t')))

    print("Number of rows:", num_rows)
    print("Number of columns:", num_cols)

def print_name_col():
    with open('04_gap-merged.tsv', 'r') as file:
        name = file.readline().strip().split('\t')
        for column_name in name:
            print(column_name)


import pandas as pd
def typeof_column():

        data = pd.read_csv('04_gap-merged.tsv', sep='\t')
        print(data.dtypes)


def get_country():
    data = pd.read_csv('04_gap-merged.tsv', sep='\t')
    countries = data['country']
    print(countries.head())


def get_last_country():
    data = pd.read_csv('04_gap-merged.tsv', sep='\t')
    countries = data['country']
    print(countries.tail())

def get_convientyear():
    data = pd.read_csv('04_gap-merged.tsv', sep='\t')
    kq = data[['country','continent','year']]
    print("5 dong dau: ")
    print(kq.head())

    print("5 dong cuoi: ")
    print(kq.tail())

#cau8
def get_first_row():
    with open('04_gap-merged.tsv', 'r') as file:
        for i in range(99):
            file.readline()
        hundred_row = file.readline().strip().split('\t')
    print ("hang thu 100 cua tap tin TSV:", hundred_row)


#cau10
def get_last_loc():
    filepath = '04_gap-merged.tsv'
    data = pd.read_csv(filepath, sep ='\t')
    last_row_label = data.index[-1]
    last_row = data.loc[last_row_label]

    print("Last Row: ")
    print(last_row)
#cau11
def firstrow_two_method():
    data = pd.read_csv('04_gap-merged.tsv', sep ='\t')
    first_row = data.iloc[0]
    row_100 = data.iloc[90]
    row_1000 = data.iloc[999]
    print('First_row: ')
    print(first_row)

    print("100 row: ")
    print(row_100)

    print("1000 row:")
    print(row_1000)
#12
def cau12():
    df = pd.read_csv('04_gap-merged.tsv', sep='\t')
    country_loc = df.loc[42,'country']
    print("Quoc gia thu 43 su dung .loc : ", country_loc)

    country_iloc = df.iloc[42]['country']
    print("Quoc gia thu 43 su dung .iloc :", country_iloc)

def cau13():
    df = pd.read_csv('04_gap-merged.tsv', sep='\t')
    rows_cols_iloc = df.iloc[[0,99,999], [0,3,5]]

    print("Cac hang va cot da chon su dung .iloc:")
    print(rows_cols_iloc)

    rows_cols_loc = df.loc[[0,99,999], ['country','lifeExp','gdpPercap']]

    print("\nCac hang va cot da chon su dung .loc:")
    print(rows_cols_loc)
def cau14():
    df = pd.read_csv('04_gap-merged.tsv', sep='\t')
    first_10_rows = df.head(10)
    print ("first 10 row of the data: ")
    print(first_10_rows)
def cau15():
    df = pd.read_csv('04_gap-merged.tsv', sep='\t')
    avr_year = df['year'].mean()
    print("Trung binh cua mot nam: ", avr_year)
def cau16():

if __name__ == '__main__':
#     # 1
#     print_hi('Nguyen Huu Toan')
#     # 2
#     odd_Number()
#
#     # 3
#     print("\nTong cac so chia het cho 2 tu 1 toi 10 : ", Tong_2())
#     Tong_3()
#
#     # 4
#     myDict()
#
#     # 5
#     My_Coureces()
#
#     # 6
#     look_ueoai()
#     # 7
#     print_a()
#     # 8
#     Given_Age()
#
#     # 9
#     input_file()
#
#     # 10
#     print("Tong 2 so a= 3 b = 5 la : ", Tonga_b(3, 5))
#     # 11
#     Matrixxx()
#     #12
#     normvecto()
#
#     #13
#     completecb()
#     #14
#     dot_product()
#     #15
#     check_matrix()
#     #16
#     rand_matrix()
#     #17
#     identity_matrix()
#     #18
#     func11()
#
#     #19
#     func12()
#     #20
#     func13()
#     #21
#     func14()
#     #22
#     #func15()
#     #23
#     func16()
#     #24
#     #func17()
#     #25
#     #func18()
#     #26
#1
    show5_line()
#2
    find_number()
#3
    print_name_col()
#4
    typeof_column()
#5
    get_country()
#6
    get_last_country()
#7
    get_convientyear()

#8
    get_first_row()

#10
    get_last_loc()
#11
    firstrow_two_method()
#12
    cau12()

    cau13()
    cau14()
    cau15()
    cau16()
###PANDAS
##