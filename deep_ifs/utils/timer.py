from time import time
time_stack = []

def tic(message=None):
    if message is not None:
        print '\t' * (len(time_stack) - 1) + message
    time_stack.append(time())

def toc(fmt="Elapsed: %.2f s"):
    try:
        print '\t' * (len(time_stack) - 1) + fmt % (time() - time_stack.pop())
    except IndexError:
        print "You have to invoke toc() before calling tic()"