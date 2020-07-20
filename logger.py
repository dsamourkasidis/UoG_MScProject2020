import os


def log(text, modelname, console=True):
    if console:
        print(text)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, 'logs-' + modelname + '.txt'))
    with open(filepath, "a") as myfile:
        myfile.write(text, "\n")