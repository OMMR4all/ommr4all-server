import openpyxl

wb_obj = openpyxl.load_workbook("/tmp/Graduale_stats.xlsx")
sheet_obj = wb_obj.active
# cell_obj = sheet_obj.cell(row=1, column=30)
# print(cell_obj.value)

for i in range(3, 2500):
    lines = 0
    skip = sheet_obj["BJ" + str(i)].value
    if skip == yes:
        continue