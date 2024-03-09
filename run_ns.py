import os
from turtle import st
import pypresence
import time

pre = pypresence.Presence(1114910688872247350)

pre.connect()

pre.update(
    state="Running NeuralNetwork",
    details="Learning | Epoches : 100/100",
    start=time.time()
)
while True:
    try:
        with os.popen("python test.py") as cmd:
            print(cmd.read())
    except Exception:
        pass
else:
    pre.close()