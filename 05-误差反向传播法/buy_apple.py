from layer_naive import MulLayer

# 正向传播
apple_price = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_amt = mul_apple_layer.forward(apple_price, apple_num)
total_amt = mul_tax_layer.forward(apple_amt, tax)

print(total_amt)

# 反向传播
# 正向传播时，是先计算苹果价格 * 苹果数量，再计算税，所以此处要先处理税
dprice = 1
dapple_amt, d_tax = mul_tax_layer.backward(dprice)
dapple_price, dapple_num = mul_apple_layer.backward(dapple_amt)
print(dapple_price, dapple_num, d_tax)

