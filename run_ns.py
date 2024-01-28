import os

while True:
    try:
        with os.popen("python test.py") as cmd:
            print(cmd.read())
    except Exception:
        pass