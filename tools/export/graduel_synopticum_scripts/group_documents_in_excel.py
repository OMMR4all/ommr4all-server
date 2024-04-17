import openpyxl
from openpyxl import Workbook
def value(i):
    if i == "#VALUE!":
        return 0
    if i == "#DIV/0!":
        return 0
    return i

wb = openpyxl.load_workbook("/tmp/Graduale_stats.xlsx", data_only=True)

# grab the active worksheet
ws = wb.get_sheet_by_name("Sheet1")
ws2 = wb.get_sheet_by_name("Minor Differences")
ws3 = wb.create_sheet("Pages_Minor Differences")
# Data can be assigned directly to cells
c_page = ""
doc_ids = []
symbols_c = 0
syl_c = 0
symbols_exist = 0
symbols_dsar = 0
symbols_hsar = 0
syl_count = 0
current_page = 3
for i in range(3, 2500):
    page = ws2['M' + str(i)].value
    if page == None:
        continue
    if page != c_page:
        current_page += 1

        ws3['A' + str(current_page)] = c_page
        ws3['B' + str(current_page)] = " ".join(doc_ids)
        ws3['C' + str(current_page)] = len(doc_ids)
        ws3['D' + str(current_page)] = symbols_c
        ws3['E' + str(current_page)] = syl_c
        ws3['F' + str(current_page)] = symbols_exist
        ws3['H' + str(current_page)] = symbols_dsar
        ws3['J' + str(current_page)] = symbols_hsar
        ws3['L' + str(current_page)] = syl_count
        doc_ids = []
        c_page = page
        symbols_c = 0
        syl_c = 0
        symbols_exist = 0
        symbols_dsar = 0
        symbols_hsar = 0
        syl_count = 0
    skip = ws2['Q' + str(i)]
    print(skip.value)
    if skip.value == 1:
        continue
    doc = ws2['A' + str(i)]
    symbols_c += value(ws2['B' + str(i)].value)
    syl_c += value(ws2['C' + str(i)].value)
    symbols_exist += value(ws2['D' + str(i)].value)
    symbols_dsar += value(ws2['F' + str(i)].value)
    print(f"Val{ws2['H' + str(i)].value}")
    symbols_hsar += value(ws2['H' + str(i)].value)
    syl_count += value(ws2['J' + str(i)].value)

    doc_ids.append(doc.value)
print("123")
# Save the file
wb.save("/tmp/Graduale_Results_05_04_2312.xlsx")
