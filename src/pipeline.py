from pprint import pprint

import utils
from models import *
from connectors import *
import mos
import pair


def main():
    utils.configure_logging()

    sourceCorpus = ZwitserloodCorpus()
    targetCorpus = UltraSuiteCorpus()

    # MOS Stuff
    
    pairs = pair.pair_speakers(
        source=sourceCorpus,
        target=targetCorpus,
        strategy=PairingStrategy.SIMILAR,
    )

    pprint(pairs)


if __name__ == "__main__":
    main()