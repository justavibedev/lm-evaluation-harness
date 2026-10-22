"""Block network socket connections in the replay child process."""

import socket


NETWORK_BLOCKED = True


def blocked(*args, **kwargs):
    raise OSError("Network disabled for the frozen-answer scoring replay")


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.getaddrinfo = blocked
socket.create_connection = blocked
