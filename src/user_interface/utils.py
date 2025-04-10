import asyncio

def run_async(func, *args, **kwargs):
    """Helper function to run async functions"""
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(func(*args, **kwargs))
    loop.close()
    return result 