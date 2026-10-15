"""The classic distributed-training foundation (M1-M2): sockets, rendezvous, collectives, DDP.

No torch imports at module level here: tests start many small processes that only need
numpy and sockets, and your laptop has limited RAM.
"""
