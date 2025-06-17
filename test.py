from concurrent.futures import ThreadPoolExecutor


def test():
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(lambda: "✅ It Works!")
        print(future.result())


test()
