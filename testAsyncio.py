import asyncio
import time


async def async_function():
    start = time.time()

    await asyncio.gather(
        greet_async("Alice"),
        greet_async("Bob"),
        greet_async("Charlie")
    )
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

async def greet_async(name):
    print(f"Hello, {name}!")
    await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(async_function())