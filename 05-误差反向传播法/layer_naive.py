class MulLayer:
    def __init__(self):
        self.x = self.y = None

    # 正向传播
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
        
    # 反向传播
    # dout为上游传过来的导数
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self) -> None:
        # pass表示什么都不做
        # 用return也行 但是用pass更清晰 表示空func
        pass
    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dy = dout

        return dx, dy