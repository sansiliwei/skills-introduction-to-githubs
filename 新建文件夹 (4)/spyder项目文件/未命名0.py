#路径为相对路径，不会报错

import csv
import matplotlib.pyplot as plt
import numpy as np


from matplotlib import font_manager as fm  

# 指定字体文件路径，确保路径正确  
font_path = 'simhei.ttf'  # Windows 下的黑体相对路径  
font_prop = fm.FontProperties(fname=font_path)  

#读取csv文件

with open('data_year.csv','r',encoding='GBK') as csv_file:
    reader=csv.reader(csv_file)
    rows = [row for row in reader]
    
#因为读取的csv文件每一行都是一个列表，而我们需要把他转换成整数，才能在作图的时候使用   
    
int_rows=[]
for row in rows:
    int_row=[]
    for val in row:
        int_row.append(int(float(val)))
    int_rows.append(int_row)    
       
  #把年份从小到大排列       
    
x=int_rows[0]
x.reverse()#reverse()不返回任何内容，只是把列表中的数字反转


y1=int_rows[1]
y1.reverse()   
 
y2=int_rows[2] 
y2.reverse()
  
y3=int_rows[3] 
y3.reverse()
  
y4=int_rows[4]
y4.reverse()
  

#fig,axes=plt.subplot(2,2),plt.subplot()返回的是单个 Axes 对象，而不是一个数组
fig,axes=plt.subplots(2,2)#plt.subplots()会返回一个 Figure 对象和一个包含多个 Axes 对象的数组
ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[1,0]
ax4=axes[1,1]





ax1.set_xlim(2014,2025)
ax1.set_ylim(137000,141800)
#下面设置一下x轴和y轴标签和刻度
ax1.set_xticks(ticks=np.arange(2014,2024,1),labels=['14','15','16','17','18','19','20','21','22','23'])
ax1.set_yticks(ticks=np.arange(137000,142000,600),labels=['137000','137600','138200','138800','139400','140000','140600','141200','141800'])
ax1.set_xlabel("年 份",fontproperties=font_prop,y=1)
ax1.set_ylabel("人 数（万人）",fontproperties=font_prop)#如果直接用汉字就会出现乱码
ax1.set_title("近十年总人口变化趋势图",fontproperties=font_prop,y=1)

ax1.plot(x,y1,color='r')

#第二个表格
ax2.set_xlim(2014,2025)
ax2.set_ylim(22000,26000)
ax2.set_xticks(ticks=np.arange(2014,2024,1),labels=['14','15','16','17','18','19','20','21','22','23'])
ax2.set_yticks(ticks=np.arange(22000,25521,440),labels=['22000','22440','22880','23320','23760','24200','24640','25080','25520'])
ax2.set_xlabel("年 份",fontproperties=font_prop,y=1)
ax2.set_ylabel("人 数（万人）",fontproperties=font_prop)#如果直接用汉字就会出现乱码
ax2.set_title("近十年 0-14岁人口变化趋势图",fontproperties=font_prop,y=1)

ax2.plot(x,y2,color='r')

#第三个表格
ax3.set_xlim(2014,2025)
ax3.set_ylim(94000,102000)
ax3.set_xticks(ticks=np.arange(2014,2024,1),labels=['14','15','16','17','18','19','20','21','22','23'])
ax3.set_yticks(ticks=np.arange(96000,101281,660),labels=['96000','96660','97320','97980','98640','99300','99960','100620','101280'])
ax3.set_xlabel("年 份",fontproperties=font_prop,y=1)
ax3.set_ylabel("人 数（万人）",fontproperties=font_prop)#如果直接用汉字就会出现乱码
ax3.set_title("近十年 15-64岁人口变化趋势图",fontproperties=font_prop,y=1)

ax3.plot(x,y3,color='r')

#第四个表格
ax4.set_xlim(2014,2025)
ax4.set_ylim(13000,23500)
ax4.set_xticks(ticks=np.arange(2014,2024,1),labels=['14','15','16','17','18','19','20','21','22','23'])
ax4.set_yticks(ticks=np.arange(13000,22000,1000),labels=['13000','14000','15000','16000','17000','18000','19000','20000','21000'])
ax4.set_xlabel("年 份",fontproperties=font_prop,y=1)
ax4.set_ylabel("人 数（万人）",fontproperties=font_prop)#如果直接用汉字就会出现乱码
ax4.set_title("近十年 65岁及以上人口变化趋势图",fontproperties=font_prop,y=1)

ax4.plot(x,y4,color='r')


#确保在运行之前定义变量 x 和 y1，y1 是你要拟合的数据点。
#考虑使用 beta[::-1] 反转系数的顺序以正确应用于 np.polyval，因为它接受的是从高次到低次的系数。
#np.linalg.inv(X.T @ X) 的计算可能会出现数值不稳定性。在实际应用中，可以考虑使用 np.linalg.pinv() 来获得伪逆

    

#下面是对未来人口的预测
   

# x.append(2024)
# 示例数据（成对的 x 和 y 数据）   
y_1 = np.array(y1)  

# 设置多项式的阶数  
n = 2# 二次多项式  
m = len(x)  

# 生成设计矩阵 X  Vandermonde矩阵在多项式拟合中特别有用
X = np.vander(x, n + 1, increasing=True)  # increasing=True 表示生成的项从 x^0 到 x^n  

# 计算 Beta  
# beta =  np.linalg.inv(X.T @ X) @ X.T @ y  
beta_1, residuals, rank, s = np.linalg.lstsq(X, y_1, rcond=None)  # 更稳定的最小二乘解  
# 输出 Beta 值  
print("计算得到的 Beta 系数:", beta_1)
# 使用numpy的向量化操作来计算多项式在x_2上的值

poly_values_1= np.polyval(beta_1[::-1], x)  # 使用numpy的polyval函数计算多项式值

def polynomial(coefficients, x):  #为了实现多项式的计算
    result = 0  
    degree = len(coefficients) - 1  
    for i, coef in enumerate(coefficients):  
        result += coef * (x ** (degree - i))  
    return result  


new_x=np.append(x,2024)
#如果这里写成new_poly_values_1=np.append(poly_values_1,polynomial(beta_1,2024))
#就会导致系数和x的相应次方不匹配，是绿线骤降
new_poly_values_1=np.append(poly_values_1,polynomial(beta_1[::-1],2024))

ax1.plot(new_x,new_poly_values_1,color='g')








# 示例数据（成对的 x 和 y 数据）   
y_2 = np.array(y2)  

# 设置多项式的阶数  
n = 2# 二次多项式  
m = len(x)  

# 生成设计矩阵 X  Vandermonde矩阵在多项式拟合中特别有用
X = np.vander(x, n + 1, increasing=True)  # increasing=True 表示生成的项从 x^0 到 x^n  

# 计算 Beta  
# beta =  np.linalg.inv(X.T @ X) @ X.T @ y  
beta_2, residuals, rank, s = np.linalg.lstsq(X, y_2, rcond=None)  # 更稳定的最小二乘解  
# 输出 Beta 值  
print("计算得到的 Beta 系数:", beta_2)
# 使用numpy的向量化操作来计算多项式在x_2上的值

poly_values_2= np.polyval(beta_2[::-1], x)  # 使用numpy的polyval函数计算多项式值

new_x=np.append(x,2024)
#如果这里写成new_poly_values_1=np.append(poly_values_1,polynomial(beta_1,2024))
#就会导致系数和x的相应次方不匹配，是绿线骤降
new_poly_values_2=np.append(poly_values_2,polynomial(beta_2[::-1],2024))
ax2.plot(new_x,new_poly_values_2,color='g')



# 示例数据（成对的 x 和 y 数据）   
y_3 = np.array(y3)  

# 设置多项式的阶数  
n = 2# 二次多项式  
m = len(x)  

# 生成设计矩阵 X  Vandermonde矩阵在多项式拟合中特别有用
X = np.vander(x, n + 1, increasing=True)  # increasing=True 表示生成的项从 x^0 到 x^n  

# 计算 Beta  
# beta =  np.linalg.inv(X.T @ X) @ X.T @ y  
beta_3, residuals, rank, s = np.linalg.lstsq(X, y_3, rcond=None)  # 更稳定的最小二乘解  
# 输出 Beta 值  
print("计算得到的 Beta 系数:", beta_3)
# 使用numpy的向量化操作来计算多项式在x_2上的值

poly_values_3= np.polyval(beta_3[::-1], x)  # 使用numpy的polyval函数计算多项式值
new_x=np.append(x,2024)
#如果这里写成new_poly_values_1=np.append(poly_values_1,polynomial(beta_1,2024))
#就会导致系数和x的相应次方不匹配，是绿线骤降
new_poly_values_3=np.append(poly_values_3,polynomial(beta_3[::-1],2024))
ax3.plot(new_x,new_poly_values_3,color='g')



# 示例数据（成对的 x 和 y 数据）   
y_4 = np.array(y4)  

# 设置多项式的阶数  
n = 2# 二次多项式  
m = len(x)  

# 生成设计矩阵 X  Vandermonde矩阵在多项式拟合中特别有用
X = np.vander(x, n + 1, increasing=True)  # increasing=True 表示生成的项从 x^0 到 x^n  

# 计算 Beta  
# beta =  np.linalg.inv(X.T @ X) @ X.T @ y  
beta_4, residuals, rank, s = np.linalg.lstsq(X, y_4, rcond=None)  # 更稳定的最小二乘解  
# 输出 Beta 值  
print("计算得到的 Beta 系数:", beta_4)
# 使用numpy的向量化操作来计算多项式在x_2上的值

poly_values_4= np.polyval(beta_4[::-1], x)  # 使用numpy的polyval函数计算多项式值
new_x=np.append(x,2024)
#如果这里写成new_poly_values_1=np.append(poly_values_1,polynomial(beta_1,2024))
#就会导致系数和x的相应次方不匹配，是绿线骤降
new_poly_values_4=np.append(poly_values_4,polynomial(beta_4[::-1],2024))
ax4.plot(new_x,new_poly_values_4,color='g')








plt.tight_layout()  #这个是真的好用，一键布局了，不用再累死累活的调整了。
plt.show()  


























  