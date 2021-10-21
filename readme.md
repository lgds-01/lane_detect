## opencv小项目——车道检测

对于视频，只要处理好每一帧的车道检测，整个视频的车道检测也相应处理成功，因此，先处理一帧的图像

### 步骤1.灰度化、滤波

```python
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# 采用高斯滤波
blur = cv2.GaussianBlur(gray,(5,5),0)
```

<img src="https://cdn.jsdelivr.net/gh/lgds-01/picture@main/uPic/image-20211021010102873.png" alt="image-20211021010102873" style="zoom: 25%;" /><img src="https://cdn.jsdelivr.net/gh/lgds-01/picture@main/uPic/image-20211021010131689.png" alt="image-20211021010131689" style="zoom:25%;" />

### 步骤2.边缘检测

```python
canny = cv2.Canny(blur,50,150)
```

<img src="https://cdn.jsdelivr.net/gh/lgds-01/picture@main/uPic/image-20211021010311379.png" alt="image-20211021010311379" style="zoom:25%;" />

### 步骤3.不规则ROI截取

```python
height,width = canny.shape[0]
# 端点根据实际测量实验得出
points = np.array([[(200, height),(800, 300),(1200, height),]],np.int32)
mask = np.zeros_like(canny)
# 产生的多边形应包含全部所需车道，但也应尽可能小，以减少噪声
cv2.fillPoly(mask,points,(255,255,255))
img = cv2.bitwise_and(canny,mask)
```

<img src="https://cdn.jsdelivr.net/gh/lgds-01/picture@main/uPic/image-20211021010502072.png" alt="image-20211021010502072" style="zoom:25%;" /><img src="/Users/lgds/Library/Application Support/typora-user-images/image-20211021010513707.png" alt="image-20211021010513707" style="zoom:25%;" />

### 步骤4.霍夫直线检测

```python
# lines包含检测到的直接的端点的坐标，shape=(n,1,4)
lines = cv2.HoughLinesP(img,1,np.pi/180,50,minLineLength=30,maxLineGap=20)
```

### 步骤5.分离左右车道

```python
# 根据斜率的正负来分离左右车道
left_lines = []
right_lines = []
for line in lines:
  for x1,y1,x2,y2 in line:
    k = (y2-y1)/(x2-x1)
    if k>0:
      right_lines.append(line)
    else:
      left_lines.append(line)
```

### 步骤6.排除异常数据

```python
# 利用各直线斜率和斜率均值的差，排除超过阈值的异常直线
def clean_lines(lines,threshold):
    ks = [(y2-y1)/(x2-x1) for line in lines for x1,y1,x2,y2 in line]
    mean = np.mean(ks)
    while len(lines) > 0:
        diff = [abs(k-mean) for k in ks]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            lines.pop(idx)
            ks.pop(idx)
        else:
            break
            
clean_lines(left_lines,0.1)
clean_lines(right_lines,0.1)
```

### 步骤7.最小二乘法拟合左右车道线

```python
def least_squares_fit(lines,ymin,ymax):
    x = [x1 for line in lines for x1,y1,x2,y2 in line]
    y = [y1 for line in lines for x1,y1,x2,y2 in line]
    x += [x2 for line in lines for x1,y1,x2,y2 in line]
    y += [y2 for line in lines for x1,y1,x2,y2 in line]
    
    # 拟合函数
    fit_fn = np.poly1d(np.polyfit(y,x,1))
    return [(int(fit_fn(ymin)),ymin),(int(fit_fn(ymax)),ymax)]
  
left = least_squares_fit(left_lines,325,height)
right = least_squares_fit(right_lines,325,height)
```

### 步骤8.画左右车道线

```python
cv2.line(frame,left[0],left[1],(0,255,0),3)
cv2.line(frame,right[0],right[1],(0,255,0),3)
```

<img src="https://cdn.jsdelivr.net/gh/lgds-01/picture@main/uPic/image-20211021012753542.png" alt="image-20211021012753542" style="zoom:25%;" />

### 步骤9.对视频的每一帧应用上述步骤

qwq,这个自食其力吧

### 项目的缺陷

1. 只能对直线车道有很好的识别率，在弯道时正确检测率很低

2. 需要人工测量修改ROI，若ROI中有多条车道，则结果偏离预期

3. 车道线缺失大部分时，检测不出该边车道

   <img src="https://cdn.jsdelivr.net/gh/lgds-01/picture@main/uPic/image-20211021093700919.png" alt="image-20211021093700919" style="zoom:25%;" /><img src="/Users/lgds/Library/Application Support/typora-user-images/image-20211021093716499.png" alt="image-20211021093716499" style="zoom:25%;" />

4. ~~项目鲁棒性不够，当画面中没有车道时会中断退出~~

5. ~~车道只有黄色和白色，因此可以分离出黄色和白色，从而可以减少误差~~
