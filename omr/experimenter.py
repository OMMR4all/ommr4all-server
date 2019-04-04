import ezodf
import argparse
import subprocess
import multiprocessing
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ods', required=True, type=str)
parser.add_argument('--run', type=str)
parser.add_argument('--script', required=True, type=str)
parser.add_argument('--additional_args', nargs="*", default=['skip_train', 'skip_predict'])
parser.add_argument('--magic_prefix', default='EXPERIMENT_OUT=')
parser.add_argument('--header_row', default=2, type=int)
parser.add_argument('--data_column_offset', default=1, type=int)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

ods = ezodf.opendoc(args.ods)

print("Spreadsheet contains %d sheet(s)." % len(ods.sheets))
for sheet in ods.sheets:
    print("-"*40)
    print("   Sheet name : '%s'" % sheet.name)
    print("Size of Sheet : (rows=%d, cols=%d)" % (sheet.nrows(), sheet.ncols()) )

experiments_sheet = [sheet for sheet in ods.sheets if sheet.name == 'experiments'][0]

rows = list(experiments_sheet.rows())

params = [cell.value for cell in rows[args.header_row]]
params_end_index = params.index('PARAMS_END')
data_column_start_index = params_end_index + 1 + args.data_column_offset
params = params[:params_end_index]

experiment_args = []
data_row_start_index = args.header_row + 1
all_rows = list(experiments_sheet.rows())
for i, row in enumerate(all_rows[data_row_start_index:]):
    ex_args = {}
    for param, cell in zip(params, row):
        if param is None or cell.value is None:
            continue

        first_data_cell = row[data_column_start_index]
        if first_data_cell.value is not None:
            # already contains data
            continue

        if isinstance(cell.value, bool):
            if cell.value:
                ex_args["--" + param] = None
        elif isinstance(cell.value, str):
            ex_args["--" + param] = cell.value.split()
        elif isinstance(cell.value, float):
            if int(cell.value) == cell.value:
                ex_args["--" + param] = int(cell.value)
            else:
                ex_args["--" + param] = cell.value
        else:
            ex_args["--" + param] = cell.value

    if len(ex_args) == 0:
        continue

    experiment_args.append({
        'row': i + data_row_start_index,
        'args': ex_args,
    })


def run(ex_args):
    cmd_list = ['python', '-u', args.script]
    if args.run:
        cmd_list = args.run.split() + cmd_list

    for key, value in ex_args.items():
        cmd_list.append(key)
        if value is not None:
            if isinstance(value, list):
                cmd_list += value
            else:
                cmd_list.append(value)

    cmd_list += ['--{}'.format(a) for a in args.additional_args]
    cmd_list = list(map(str, cmd_list))

    process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)
    result = None
    while True:
        output = process.stdout.readline().decode('utf-8').strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            if output.startswith(args.magic_prefix):
                result = output[len(args.magic_prefix):]
                result = result.split(',')
            else:
                if args.verbose:
                    print(output)

    rc = process.poll()

    return result


if len(experiment_args) == 0:
    print("Nothing to compute found")
else:
    with multiprocessing.Pool(processes=len(experiment_args)) as p:
        ex_results = list(tqdm(p.imap(run, [ex_args['args'] for ex_args in experiment_args]), total=len(experiment_args)))

    for ex_args, ex_result in zip(experiment_args, ex_results):
        data_row = ex_args['row']
        row = all_rows[data_row]
        for i, value in enumerate(ex_result):
            cell = row[i + data_column_start_index]
            cell.set_value(value)

    ods.save()
