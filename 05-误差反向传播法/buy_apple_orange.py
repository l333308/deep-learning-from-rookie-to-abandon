from layer_naive import *

ap_price = 100
ap_num = 2
pr_price = 150
or_num = 3
tax = 1.1

# 正向传播
ap_amt_layer = MulLayer()
or_amt_layer = MulLayer()
ap_or_amt_layer = AddLayer()
tax_layer = MulLayer()

ap_amt = ap_amt_layer.forward(ap_price, ap_num)
or_amt = or_amt_layer.forward(pr_price, or_num)
ap_or_amt = ap_or_amt_layer.forward(ap_amt, or_amt)
total_amt = tax_layer.forward(ap_or_amt, tax)
print("税后总金额：", total_amt)

# 反向传播
d_total_amt = 1
d_ap_or_amt, d_tax = tax_layer.backward(d_total_amt)
d_ap_amt, d_or_amt = ap_or_amt_layer.backward(d_ap_or_amt)
d_ap_price, d_ap_num = ap_amt_layer.backward(d_ap_amt)
d_or_price, d_or_num = or_amt_layer.backward(d_or_amt)

print("\n苹果数量：", d_ap_num) # 110
print("苹果单价：", d_ap_price) # 2.2
print("橘子单价：", d_or_price) # 3.3
print("橘子数量：", d_or_num) # 165
print("消费税", d_tax) #650
