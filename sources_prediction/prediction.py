import sys
from datetime import datetime

from tqdm.contrib.concurrent import process_map

import helpers

DEBUG = False
CV = True


def main():  # pragma: no cover
    dataset = helpers.load_dataset()
    df = helpers.prepare(dataset)
    gb = (
        df.groupby("LogicFile")
            .filter(lambda x: len(x) > 2000)
            .groupby("LogicFile")
    )
    results = process_map(predict_group, gb, chunksize=1, max_workers=20)
    current_ts = 1647338205  # time.time()
    group_to_mape = {}
    for group_name, result, mape in results:
        if DEBUG:
            print("%r predicted: %s" % (group_name, result))
        group_to_mape[group_name] = mape
        if result is None:
            print("No prediction for %r" % (group_name), file=sys.stderr)
        elif result < current_ts:
            date = datetime.fromtimestamp(result).strftime("%Y-%m-%d %X")
            print(
                "%r predicted in the past: %s" % (group_name, date),
                file=sys.stderr,
            )
    if CV:
        print("Groups mape are %r" % group_to_mape)


def predict_group(*args):
    group_name = args[0][0]
    group = args[0][1]
    try:
        m = helpers.fit(group)
        result = helpers.forecast(m, group_name, DEBUG)
        # helpers.save_model(m, group_name)
        mape = None
        if CV:
            mape = helpers.cv(m, group, DEBUG)
        return group_name, result, mape
    except Exception as exc:
        print(
            "%r generated an exception: %s" % (group_name, exc),
            file=sys.stderr,
        )
        return group_name, None, None


if __name__ == "__main__":
    main()
