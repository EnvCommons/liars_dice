from openreward.environments import Server
from env import LiarsDiceEnvironment

if __name__ == "__main__":
    server = Server([LiarsDiceEnvironment])
    server.run()
