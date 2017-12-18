from time import time
time_stack = []
logfile = None


def tic(message=None):
    """
    Start counting
    :param message: additional message to print to logfile and stdout
    """
    time_stack.append(time())
    if message:
        output = '\t' * (len(time_stack) - 1) + str(message)
        print output
        if logfile is not None:
            logfile.write(output + '\n')


def log(message):
    """
    Write message to logfile and stdout
    :param message: message to print
    :return: 
    """
    n_tabs = (len(time_stack) - 1)
    output = '\t' * n_tabs + str(message)
    print output
    if logfile is not None:
        logfile.write(output + '\n')


def toc(message=None):
    """
    Stop counting
    :param message: additional message to print to logfile and stdout
    """
    fmt = 'Elapsed: %.2f s'
    n_tabs = (len(time_stack) - 1)
    try:
        output = '\t' * n_tabs + fmt % (time() - time_stack.pop())
        if message:
            output = '\t' * n_tabs + str(message) + '\n' + output
        print output
        if logfile is not None:
            logfile.write(output + '\n')
        print ''
    except IndexError:
        print "You have to invoke toc() before calling tic()\n"


def setup_logging(filename):
    """
    Setup a logfile
    :param filename: where to write logs
    """
    global logfile
    logfile = open(filename, 'w', 0)
