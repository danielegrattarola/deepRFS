from time import time
time_stack = []

def tic(message=None):
    time_stack.append(time())
    if message is not None:
        print '\t' * (len(time_stack) - 1) + message

def toc(fmt="Elapsed: %.2f s"):
    try:
        print '\t' * (len(time_stack) - 1) + fmt % (time() - time_stack.pop())
    except IndexError:
        print "You have to invoke toc() before calling tic()"