"""
Some usefull fuction decorators
"""

import time

# Decorator to insure a function is thread safe
def threadsafe(lock):
    def wrapper1(func):
        def wrapper(*args, **kwargs):
            lock.acquire()
            func(*args, **kwargs)
            lock.release()
        return wrapper
    return wrapper1


# Decorator to insure a function is thread safe
# Including timeing
def threadsafetimer(lock):
    start = time.time()
    def wrapper1(func):
        def wrapper(*args, **kwargs):
            lock.acquire()
            func(*args, **kwargs)
            lock.release()
        return wrapper
    print(time.time() - start)
    return wrapper1