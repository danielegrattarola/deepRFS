from time import time
time_stack = []
logfile = None


def tic(message=None):
    time_stack.append(time())
    if message:
        output = '\t' * (len(time_stack) - 1) + str(message)
        print output
        if logfile is not None:
            logfile.write(output + '\n')


def log(message):
    n_tabs = (len(time_stack) - 1)
    output = '\t' * n_tabs + str(message)
    print output
    if logfile is not None:
        logfile.write(output + '\n')


def toc(message=None):
    fmt = 'Elapsed: %.2f s'
    n_tabs = (len(time_stack) - 1)
    try:
        output = '\t' * n_tabs + fmt % (time() - time_stack.pop())
        print output
        if logfile is not None:
            logfile.write(output + '\n')
        if message:
            output = '\t' * n_tabs + str(message)
            print output
            if logfile is not None:
                logfile.write(output + '\n')
        print ''
    except IndexError:
        print "You have to invoke toc() before calling tic()\n"


def setup_logging(logpath):
    global logfile
    logfile = open(logpath, 'w', 0)
