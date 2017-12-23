"""
Some usefull fuction decorators
"""


# Decorator to insure a function is thread safe
def threadsafe(lock):
    def wrapper1(func):
        def wrapper(*args, **kwargs):
            lock.acquire()
            func(*args, **kwargs)
            lock.release()
        return wrapper
    return wrapper1
