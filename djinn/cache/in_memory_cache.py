__simple_cache = {}

def cache(func):
    def wrapper(*arg, **kwargs):
        func_id = id(func)
        global __simple_cache
        if func_id in __simple_cache:
            return __simple_cache[func_id]
        else:
            result = func(*arg, **kwargs)
            __simple_cache[func_id] = result
            return result
    return wrapper