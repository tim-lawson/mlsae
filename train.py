from simple_parsing import parse

from mlsae.trainer import RunConfig, train

if __name__ == "__main__":
    train(parse(RunConfig))
