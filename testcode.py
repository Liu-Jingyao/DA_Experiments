import torch
import transformers

if __name__ == '__main__':
    from queue import PriorityQueue as PQ

    pq = PQ()
    pq.put((1, 'a'))
    pq.put((2, 'c'))
    pq.put((2, 'b'))
    pq.put((2, 'b'))
    print(pq.queue)  # [(1, 'a'), (2, 'b'), (2, 'b'), (2, 'c')]
    item0 = pq.get()  # (1, 'a')
    print(pq.queue)  # [(2, 'b'), (2, 'b'), (2, 'c')]

    print(pq.qsize())  # 优先队列的尺寸

    while not pq.empty():
        print(pq.get())