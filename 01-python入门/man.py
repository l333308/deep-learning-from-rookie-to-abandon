class Man:
    def __init__(self, name) -> None:
        self.name = name
        print("initialized...")

    def hello(self):
        print("hello, " + self.name + "!")

    def goodbye(self):
        print("good-bye, " + self.name + "!")


m = Man("pikaka")
m.hello()
m.goodbye()