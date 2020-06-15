import sys
sys.path.insert(0, ".")
from camembert.downstream.finetune.env.imports import gspread, ServiceAccountCredentials, os
from camembert.downstream.finetune.io_.logger import printing
from camembert.downstream.finetune.env.dir.project_directories import CLIENT_GOOGLE_CLOUD, SHEET_NAME_DEFAULT, TAB_NAME_DEFAULT


# use creds to create a client to interact with the Google Drive API

SCOPES_GOOGLE_SHEET = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

REPORTED_VARIABLE_PER_SHEET = {"experiments_tracking": {"git_id": 1,
                                                        "job_id": 2,
                                                        "tasks": 3,
                                                        "description": 4,
                                                        "logs_dir": 5,
                                                        "target_dir": 6,
                                                        "env": 7,
                                                        "completion": 8,
                                                        "evaluation_dir": 9,
                                                        "tensorboard_dir": 10}
                               }

try:
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(CLIENT_GOOGLE_CLOUD, 'client.json'), SCOPES_GOOGLE_SHEET)
    # Extract and print all of the values
    SHEET_NAME_DEFAULT, TAB_NAME_DEFAULT = "model_evaluation", "experiments_tracking"
except Exception as e:
    creds = None
    print("REPORTING : google sheet reporting not available {}".format(e))


def open_client(credientials=creds, sheet_name=SHEET_NAME_DEFAULT, tab_name=TAB_NAME_DEFAULT):
    client = gspread.authorize(credientials)
    sheet = client.open(sheet_name)
    sheet = sheet.worksheet(tab_name)
    return sheet, sheet_name, tab_name


def append_reporting_sheet(git_id, tasks, rioc_job, description, log_dir, target_dir, env, status,
                           verbose=1):
    sheet, sheet_name, tab_name = open_client()
    # Find a workbook by name and open the first sheet
    # Make sure you use the right name here.
    #worksheet_list = sheet.worksheets()
    if not rioc_job.startswith("local"):
        sheet.append_row([git_id,  rioc_job, tasks, description, log_dir, target_dir, env, status, None, None, None, None,"-"])
        list_of_hashes = sheet.get_all_records()
        printing("REPORT : Appending report to page {} in sheet {} of {} rows and {} columns ",
                 var=[tab_name, sheet_name, len(list_of_hashes)+1, len(list_of_hashes[0])],
                 verbose=verbose,
                 verbose_level=1)
    else:
        print("LOCAL env not updating sheet")
        list_of_hashes = ["NOTHING"]
    return len(list_of_hashes)+1, len(list_of_hashes[0])


def update_status(row, value, col_number=8, sheet=None, verbose=1):
    if sheet is None:
        sheet, sheet_name, tab_name = open_client()
    if value is not None:
        sheet.update_cell(row, col_number, value)
        printing("REPORT : col {} updated in sheet with {} ", var=[col_number, value], verbose=verbose, verbose_level=1)


git_id = "aaa"
rioc= "XXX"
description = "test"
log_dir = "--"
target_dir = "--"
env = "rioc"
completion = "running"
