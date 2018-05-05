#coding:utf-8

import  os
import numpy as np
import matplotlib
import matplotlib.pyplot as chart

matrix0 = []
matrix1 = []
matrix2 = []
matrix3 = []
matrix4 = []
matrix5 = []
matrix6 = []
matrix7 = []
matrix8 = []
matrix9 = []
numb = 0  # 记录总个数
filename = 'magic04.txt'

def fun_a(numb):
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            numb += 1

            if not lines:
                break

            mat0_tmp, mat1_tmp, mat2_tmp, \
            mat3_tmp, mat4_tmp, mat5_tmp, \
            mat6_tmp, mat7_tmp, mat8_tmp, \
            mat9_tmp, attr = [i for i in lines.split(",")]
            matrix0.append(float(mat0_tmp))
            matrix1.append(float(mat1_tmp))
            matrix2.append(float(mat2_tmp))
            matrix3.append(float(mat3_tmp))
            matrix4.append(float(mat4_tmp))
            matrix5.append(float(mat5_tmp))
            matrix6.append(float(mat6_tmp))
            matrix7.append(float(mat7_tmp))
            matrix8.append(float(mat8_tmp))
            matrix9.append(float(mat9_tmp))

        print sum(matrix0) / numb
        print sum(matrix1) / numb
        print sum(matrix2) / numb
        print sum(matrix3) / numb
        print sum(matrix4) / numb
        print sum(matrix5) / numb
        print sum(matrix6) / numb
        print sum(matrix7) / numb
        print sum(matrix8) / numb
        print sum(matrix9) / numb

    y = [matrix0, matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9]
    print np.cov(y)
    a = np.array(matrix0)
    b = np.array(matrix1)
    chart.plot(a, b, 'yo')
    chart.show()

    data = a
    mean = data.mean()
    std = data.std()
    x = np.arange(-200, 200, 0.1)
    y = normal_fun(x, mean, std)
    chart.plot(x, y, color='#054E9F')
    chart.hist(data, color='#F54515', bins=10, rwidth=0.9, normed=True)
    chart.title('Data distribution')
    chart.xlabel('Data')
    chart.ylabel('Probability')
    chart.show()

def normal_fun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

if __name__ == '__main__':
    fun_a(numb)
def variancefun(a):
    array = np.array(a)
    var = array.var()
    return var
print variancefun(matrix0)
print variancefun(matrix1)
print variancefun(matrix2)
print variancefun(matrix3)
print variancefun(matrix4)
print variancefun(matrix5)
print variancefun(matrix6)
print variancefun(matrix7)
print variancefun(matrix8)
print variancefun(matrix9)
def covfun(a):
    array = np.array(a)
    cov = np.cov(array)
    return cov
print covfun(matrix0)
print covfun(matrix1)
print covfun(matrix2)
print covfun(matrix3)
print covfun(matrix4)
print covfun(matrix5)
print covfun(matrix6)
print covfun(matrix7)
print covfun(matrix8)
print covfun(matrix9)