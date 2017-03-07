from time import time
time_stack = []

def tic(message=None):
    time_stack.append(time())
    if message is not None:
        print '\t' * (len(time_stack) - 1) + message

def toc(message=None):
    fmt = 'Elapsed: %.2f s'
    n_tabs = (len(time_stack) - 1)
    try:
        print '\t' * n_tabs + fmt % (time() - time_stack.pop())
        if message:
            print '\t' * n_tabs + message
        print ''
    except IndexError:
        print "You have to invoke toc() before calling tic()"