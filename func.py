import datetime, timeit
    
def get_timestamp():
    timestamp = datetime.datetime.now().isoformat(' ', 'minutes')  # without microseconds
    timestamp = timestamp.replace(' ', '-').replace(':', '-')
    return timestamp