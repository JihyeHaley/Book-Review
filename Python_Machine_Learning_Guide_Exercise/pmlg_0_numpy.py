import numpy as np # 관용적 표현 
import pandas as pd

# number!!!! thtz numpy
array1 = np.array([1, 2, 3, 4])
print(f'array 1type: {type(array1)}') # numpy.ndarray
print(f'array 1 shape: {array1.shape}') # 1차원: (4, )


array2 = np.array([[1, 2, 3], [2, 3, 4]])
print(f'array 2type: {type(array2)}')         # (행,열
print(f'array 2 shape: {array2.shape}') # 2차원 (2, 3)

array3 = np.array([[1, 2, 3]])
print(f'array 2type: {type(array3)}')
print(f'array 2 shape: {array3.shape}') # 2차원

print(f'array1: {array1.ndim}차원, array2: {array2.ndim}차원, array3: {array3.ndim}차원')


'''
data type 
    * int형 (8bit, 16bit, 32bit)
    * unsigned int형(8bit, 16bit, 32bit)
    * float형 (16bit, 32bit, 64bit, 128bit)
    * complex타입


    ndarray내의 데이터 타입은 같은 데이터 타입만 가능 
    e.g. int/float 함께 있을 수 없다. -> 확인은 `.dtype`으로 가능
'''

list1 = [1, 2, 3]
print(type(list1)) # list
array1 = np.array(list1) # ndarray로 변경
print(type(array1)) # numpy.ndarray
print(array1, array1.dtype) # [1, 2, 3] int64

list2 = [1, 2, 'test']
array2 = np.array(list2) 
print(array2, array2.dtype) # 숫자와 글자가 결합되어 있음. 유니코드 문자열 값으로변환 -> <U21

list3 = [1, 2, 3.0]
array3 = np.array(list3) 
print(array3, array3.dtype) # 정수와 실수가 결합되어 있음. 실수형으로 변환 -> float64

'''
파이썬 기반의 머신러닝 알고리즘은 대부분 메모리로 데이터를 전체 로딩한 다음 이를 기반으로 알고리즘을 적용하기 때문에 
대용량의 데이터를 로딩할 때는 수행속도가 느려지거나 메모리 부족으로 오류가 발생할 수 있다. 
-> float라면 int형으로 바꿔서 메모리를 절약하자
'''
print('*'*40, '\n')
array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2 = array_float.astype('int32')
print(array_int2, array_int2.dtype)

'''
ndarray를 편리하게, arange, zeros, ones
    * arange(range랑 비슷)
        - array를 range()로 표현..  0부터 함수 인자값 -1까지 순차적으로 ndarray의 데이터값으로 변환
'''

print('*'*40, '\n')

sequence_array = np.arange(10)
print(type(sequence_array))
print(list(sequence_array))
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)

'''
    * azeros
        - default 함수 인자는 stop값이며, 0부터 stop값이 10에서 -1을 더한 9까지의 연속 숫자값으로 구성된 1차원 ndarray를 만들어 줍니다. 여기서 stop값만 부여했으나 range와 유사하게 start 값도 부여해 0이 아닌 다른 값부터 시작한 연속 값을 부여할 수도 있습니다.
'''
print('*'*40, '\n')

zero_array = np.zeros((3, 2), dtype='int32')
print(type(zero_array))
print(list(zero_array))
print(zero_array.dtype, zero_array.shape)


'''
    * ones
        - dtype을 새롭게 지정하지 않으면, flaot 타입으로 나온다.
'''
one_array = np.ones((3,2), dtype='int32')
print(one_array)
print(one_array.dtype, one_array.shape)


'''
reshape()

reshape()메서드는 ndarrya를 특정 차원 및 크기로 변환합니다.
변환을 원하는 크를 함수 인자로 부여하면 된다. 

c.f. 조금 더 효율적으로 사용하기 위해서는 reshape()에 -1값을 인자로 적용해보면 된다.
'''

array1 = np.arange(10)
print(f'array1:\n {array1}')

array2 = array1.reshape(2, 5)
print(array2)
print(f'array2:\n {array2.shape}')

array3 = array1.reshape(5, 2)
print(array3)
print(f'array3:\n {array3.shape}')


array1 = np.arange(10)
print(array1)
print('$'*40)
array2 = array1.reshape(-1, 5)
print(array2)
# print(array2.shape)
array3 = array1.reshape(5, -1)
print(array3)
# print(array3.shape)

print('3d')
array1 = np.arange(8)
array3d = array1.reshape((2, 2, 2))
# print(array3d, array3d.ndim)
# print(array3d.tolist())

print('$'*40)
# 3차원 ndarray를 2차원 ndarray로 변환
array5 = array3d.reshape(-1, 1)
print('array5:\n', array5)
print('array5 shape:', array5.shape)
print('array5 tolist():', array5.tolist())
print(array5, array5.ndim)

print('$'*40)
# 1차원 ndarray를 2차원 ndarray로 반환
array6 = array1.reshape(-1, 1)
print('array6:\n', array6.tolist())
print('array6 shape:', array6.shape)
print(array6, array6.ndim)

array7 = array1.reshape(1,-1)
print(array7)


'''
Indexing (ndarray의 데이터 세트 선택하기)
1. 특정 데이터만 추출: 원하는 위치의 인덱스 값을 지정하면 해당 위치의 데이터가 반환됩니다.
2. 슬라이싱(slicing): 슬라이싱은 연속도니 인덱스상의 ndarray를 추출하는 방식입니다.
':' 기호 사이에 시작 인덱스와 종료 인덱스를 표시하면 시작 인덱스에서 종료 인덱스 -1 위치에 있는 데이터의 ndarray를 반환합니다. 
e.g. [1:5] 라고 하면, 시작 인덱스 1과 종료 인덱스 4까지에 해당하는 ndarray를 반환합니다.
3. 팬시 인덱싱(Fancy indexing): 일정한 인덱싱 집합을 리스트 또는 ndarray형태로 지정해 해당 위치에 있는 데이터의 ndarray를 반환합니다.
4. 불린 인덱싱(Boolean indexing): 특정 조건에 해당하는지 여부인 True/False 값 인덱싱 집합을 기반으로 True에 해당하는 인덱스의 위치에 있는 ndarray를 반환한다.
'''

print('%'*60)
print('Indexing 단일값 추출 (1차원)')
# 1부터 9까지의 1차원 ndarray 생성
array1 = np.arange(start=1, stop=10) # [1 2 3 4 5 6 7 8 9]
# array2 = np.arange(1, 10) # [1 2 3 4 5 6 7 8 9]
print('array1:', array1)
# print('array2:', array2)

# index는 0부터 시작하므로 array1[2]는 3번째 index 위치의 데이터값을 의미한다.
value = array1[2]
print('value: ', value) # 3
print(type(value)) # numpy.int32
'''나오는 타입의 형태만을 잘 보도록! 리스트랑 쓰는거 똑같음'''

print('\n%'*60)
print('Indexing 단일값 추출 (2차원 이상)')

array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(-1, 3)
print(array2d)
# print(array2d.ndim)  # 2차원

print(f'(row=0, col=0) index가르키는 값: {array2d[0, 0]}') # 1 
print(f'(row=0, col=1) index가르키는 값: {array2d[0, 1]}') # 2
print(f'(row=1, col=0) index가르키는 값: {array2d[1, 0]}') # 4
print(f'(row=2, col=2) index가르키는 값: {array2d[2, 2]}') # 9


print(f'(array2d[0:2, 0:2] {array2d[0:2, 0:2]}')
print(f'(array2d[1:3, 0:3] {array2d[1:3, 0:3]}')
print(f'(array2d[1:3, :] {array2d[1:3, :]}')
print(f'(array2d[:, :] {array2d[:, :]}')
print(f'(array2d[:2, 1:] {array2d[:2, 1:]}')
print(f'(array2d[:2, 0] {array2d[:2, 0]}')


'''내일 오전부터는 팬시 인덱싱을 해보자!!'''