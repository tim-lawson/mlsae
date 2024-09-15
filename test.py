from simple_parsing import parse

from mlsae.trainer import RunConfig, test

if __name__ == "__main__":
    test(parse(RunConfig))
