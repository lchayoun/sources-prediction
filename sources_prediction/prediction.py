import sys

from prophet import Prophet

import helpers


def main():  # pragma: no cover
    dataset = helpers.load_dataset()
    df = helpers.prepare(dataset)
    gb = df.groupby('LogicFile')
    for group_name, group in gb:
        m = Prophet()
        try:
            helpers.fit(m, group)
            helpers.forecast(m, group_name)
            # helpers.save_model(m, group_name)
            # helpers.cv(m, group)
        except Exception as inst:
            print('Failed predicting ' + group_name + ':', inst,
                  file=sys.stderr)


if __name__ == "__main__":
    main()
