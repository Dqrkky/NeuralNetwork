import os

while True:
    with os.popen("python test.py") as cmd:
        print(cmd.read())